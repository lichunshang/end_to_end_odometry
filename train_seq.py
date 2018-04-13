import data_roller as data
import config
import tools
import os
import datetime
import model
import simple_model
import losses
import tensorflow as tf
import numpy as np
import time

# =================== CONFIGURATIONS ========================
#cfg = config.SeqTrainConfigs
cfg = config.SeqTrainConfigsSmallSteps
val_cfg = config.SeqTrainConfigsSmallStepsValidation
config.print_configs(cfg)

lr_set = 0.001
lr_schedule = {
    0: 0.0001,
    40: 0.0001,
    70: 0.00005
}
start_epoch = 0
# alpha_schedule = {0: 0.99,  # epoch: alpha
#                   20: 0.9,
#                   40: 0.5,
#                   60: 0.1,
#                   80: 0.025}
alpha_schedule = {0: 1,
                  5: 0.99,
                  15: 0.9,
                  25: 0.8}

tensorboard_meta = True

# =================== MODEL + LOSSES + Optimizer ========================
# inputs, lstm_initial_state, initial_poses, is_training, fc_outputs, se3_outputs, lstm_states = model.build_seq_model(
#     cfg)
# se3_labels, fc_labels = model.model_labels(cfg)
with tf.device("/gpu:0"):
    inputs, lstm_initial_state, initial_poses, is_training, fc_outputs, se3_outputs, lstm_states = simple_model.build_seq_model(
        cfg)
    se3_labels, fc_labels = simple_model.model_labels(cfg)

# build model for validation
# with tf.device("/gpu:1"):
#     val_inputs, val_lstm_init, val_init_poses, _, val_fc_outputs, val_se3_outputs, val_lstm_states = simple_model.build_seq_model(
#         val_cfg)
#     val_se3_labels, val_fc_labels = simple_model.model_inputs(val_cfg)

tools.printf("Building losses...")
with tf.device("/gpu:0"):
    with tf.variable_scope("Losses"):
        se3_losses, se3_xyz_losses, se3_quat_losses = losses.se3_losses(se3_outputs, se3_labels, cfg.k_se3)
        fc_losses, fc_xyz_losses, fc_ypr_losses, \
        x_loss, y_loss, z_loss = losses.pair_train_fc_losses(fc_outputs, fc_labels, cfg.k_fc)
        alpha = tf.placeholder(tf.float32, name="alpha", shape=[])  # between 0 and 1, larger favors fc loss
        total_losses = (1 - alpha) * se3_losses + alpha * fc_losses

tools.printf("Building optimizer...")
with tf.variable_scope("Optimizer"):
    # dynamic learning rates
    lr = tf.placeholder(tf.float32, name="se3_lr", shape=[])
    trainer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_losses, colocate_gradients_with_ops=False)

# ================ LOADING DATASET ===================

tools.printf("Loading training data...")
#train_data_gen = data.StatefulDataGen(cfg, "/home/cs4li/Dev/KITTI/dataset/",
#                                      ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"])
train_data_gen = data.StatefulRollerDataGen(cfg, "/home/cs4li/Dev/KITTI/dataset/", ["00"], frames=None)
tools.printf("Loading validation data...")
val_data_gen = data.StatefulRollerDataGen(cfg, "/home/cs4li/Dev/KITTI/dataset/", ["10"], frames=[range(500)])


# for evaluating validation loss
global_validation_lstm_states = np.zeros([2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size])
global_

def calc_val_loss(sess, i_epoch, losses_log):
    curr_lstm_states = np.zeros([2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size])

    init_poses = np.zeros([cfg.batch_size, 7], dtype=np.float32)
    init_poses[:, 3] = np.ones([cfg.batch_size], dtype=np.float32)

    val_data_gen.next_epoch()

    while val_data_gen.has_next_batch():
        reset_state, batch_data, _, se3_ground_truth = val_data_gen.next_batch()

        init_poses = se3_ground_truth[0, :, :]

        curr_lstm_states = data.reset_select_lstm_state(curr_lstm_states, reset_state)

        _se3_outputs, _se3_losses, _curr_lstm_states = sess.run(
            [se3_outputs, se3_losses, lstm_states, ],
            feed_dict={
                inputs: batch_data,
                lstm_initial_state: curr_lstm_states,
                initial_poses: init_poses,
                se3_labels: se3_ground_truth[1:, :, :],
                is_training: False
            }
        )

        curr_lstm_states = _curr_lstm_states
#        init_poses = _se3_outputs[cfg.sequence_stride, :, :]
        losses_log.log(i_epoch, val_data_gen.curr_batch() - 1, _se3_losses)

    return losses_log


# =================== SAVING/LOADING DATA ========================
results_dir_path = tools.create_results_dir("train_seq")
tf_checkpoint_saver = tf.train.Saver(max_to_keep=3)
tf_best_saver = tf.train.Saver(max_to_keep=2)

tf_restore_saver = tf.train.Saver()
restore_model_file = None
restore_model_file = "/home/cs4li/Dev/end_to_end_visual_odometry/results/train_seq_20180412-19-02-06/best_val/model_best_val_checkpoint-1"

# just for restoring pre trained cnn weights
cnn_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "^cnn_layer.*")
cnn_init_tf_saver = tf.train.Saver(cnn_variables)
cnn_init_model_file = None
# cnn_init_model_file = "/home/cs4li/Dev/end_to_end_visual_odometry/results/" \
#                       "flownet_weights/flownet_s_weights"

# =================== TRAINING ========================
# config = tf.ConfigProto(allow_soft_placement=True)

tf.summary.scalar("training_loss", total_losses)
tf.summary.scalar("fc_losses", fc_losses)
tf.summary.scalar("se3_losses", se3_losses)
tf.summary.scalar("fc_xyz_losses", fc_xyz_losses)
tf.summary.scalar("fc_ypr_losses", fc_ypr_losses)
tf.summary.scalar("se3_xyz_losses", se3_xyz_losses)
tf.summary.scalar("se3_quat_losses", se3_quat_losses)
tf.summary.scalar("x_loss", x_loss)
tf.summary.scalar("y_loss", y_loss)
tf.summary.scalar("z_loss", z_loss)

early_activations = tf.get_collection(tf.GraphKeys.ACTIVATIONS, "^cnn_layer/cnn_over_timesteps/cnn_model")

merged_summary_op = tf.summary.merge_all()

with tf.Session(config=None) as sess:
    if cnn_init_model_file:
        tools.printf("Taking initialization weights from %s..." % cnn_init_model_file)
        sess.run(tf.global_variables_initializer())
        cnn_init_tf_saver.restore(sess, cnn_init_model_file)
    elif restore_model_file:
        tools.printf("Restoring model weights from %s..." % restore_model_file)
        tf_restore_saver.restore(sess, restore_model_file)
    else:
        tools.printf("Initializing variables...")
        sess.run(tf.global_variables_initializer())

    # Visualization
    if tensorboard_meta:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    else:
        run_options = None
        run_metadata = None
    writer = tf.summary.FileWriter('graph_viz' + "_%s" % datetime.datetime.today().strftime('%Y%m%d-%H-%M-%S') + '/')
    writer.add_graph(tf.get_default_graph())
    writer.flush()

    # Set up for training
    total_batches = train_data_gen.total_batches()
    train_losses_log_set = tools.LossesSet(
        ["total", "fc", "se3", "fc_xyz", "fc_ypr", "se3_xyz", "se3_quat", "x", "y", "z"],
        cfg.num_epochs, train_data_gen.total_batches())
    val_losses_log = tools.Losses("se3_val", cfg.num_epochs, val_data_gen.total_batches())

    tools.printf("Start training loop...")
    tools.printf("lr: %f" % lr_set)
    tools.printf("start_epoch: %f" % start_epoch)
    tools.printf("alpha_schedule: %s" % alpha_schedule)

    best_val_loss = 9999999999
    alpha_set = -1
    i_epoch = 0

    last_epoch_lstm_states = np.zeros([2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size])
    for i_epoch in range(start_epoch, cfg.num_epochs):
        tools.printf("Training Epoch: %d ..." % i_epoch)

        curr_lstm_states = np.zeros([2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size])
        if (i_epoch > 0):
            mid = cfg.batch_size/2
            mid = int(mid)
            curr_lstm_states[0, :, 1:mid, :] = last_epoch_lstm_states[0][:, 0:(mid-1), :]
            curr_lstm_states[0, :, mid+1:, :] = last_epoch_lstm_states[0][:, mid:-1, :]
            curr_lstm_states[1, :, 1:mid, :] = last_epoch_lstm_states[1][:, 0:(mid - 1), :]
            curr_lstm_states[1, :, mid + 1:, :] = last_epoch_lstm_states[1][:, mid:-1, :]

        start_time = time.time()

        if i_epoch in alpha_schedule.keys():
            alpha_set = alpha_schedule[i_epoch]
            tools.printf("alpha set to %f" % alpha_set)

        if i_epoch in lr_schedule.keys():
            lr_set = lr_schedule[i_epoch]
            tools.printf("learning rate set to %f" % lr_set)

        init_poses = np.zeros([cfg.batch_size, 7], dtype=np.float32)
        init_poses[:, 3] = np.ones([cfg.batch_size], dtype=np.float32)

        while train_data_gen.has_next_batch():
            j_batch = train_data_gen.curr_batch()
            # get inputs
            reset_state, batch_data, \
            fc_ground_truth, se3_ground_truth = train_data_gen.next_batch()

            curr_lstm_states = data.reset_select_lstm_state(curr_lstm_states, reset_state)
            #init_poses = data.reset_select_init_pose(init_poses, reset_state)
            #init_poses = np.zeros([cfg.batch_size, 7], dtype=np.float32)
            #init_poses[:, 3] = np.ones([cfg.batch_size], dtype=np.float32)

            #shift se3 ground truth to be relative to the first pose
            init_poses = se3_ground_truth[0,:,:]

            # Run training session
            _, _curr_lstm_states, _se3_outputs, summary, _total_losses = sess.run(
                [trainer, lstm_states, se3_outputs, merged_summary_op, total_losses],
                feed_dict={
                    inputs: batch_data,
                    se3_labels: se3_ground_truth[1:,:,:],
                    fc_labels: fc_ground_truth,
                    lstm_initial_state: curr_lstm_states,
                    initial_poses: init_poses,
                    lr: lr_set,
                    alpha: alpha_set,
                    is_training: True,
                },
                options=run_options,
                run_metadata=run_metadata
            )
            curr_lstm_states = _curr_lstm_states
            #init_poses = _se3_outputs[cfg.sequence_stride, :, :]

            # for tensorboard
            if tensorboard_meta:
                writer.add_run_metadata(run_metadata, 'epochid=%d_batchid=%d' % (i_epoch, j_batch))
                writer.add_summary(summary, i_epoch * train_data_gen.total_batches() + j_batch)

            # print stats
            tools.printf("batch %d/%d: Loss:%.7f" % (
                train_data_gen.curr_batch(), train_data_gen.total_batches(),
                _total_losses))
        last_epoch_lstm_states = curr_lstm_states

        tools.printf("Evaluating validation loss...")
        val_losses_log = calc_val_loss(sess, i_epoch, val_losses_log)
        ave_val_loss = val_losses_log.get_ave(i_epoch)

        # check for best results
        if ave_val_loss < best_val_loss:
            tools.printf("Saving best result...")
            best_val_loss = ave_val_loss
            train_losses_log_set.write_to_disk(results_dir_path)
            val_losses_log.write_to_disk(results_dir_path)
            tf_best_saver.save(sess, os.path.join(results_dir_path, "best_val", "model_best_val_checkpoint"),
                               global_step=i_epoch)
            tools.printf("Best val loss, model saved.")
        if i_epoch % 5 == 0:
            tools.printf("Saving checkpoint...")
            train_losses_log_set.write_to_disk(results_dir_path)
            val_losses_log.write_to_disk(results_dir_path)
            tf_checkpoint_saver.save(sess, os.path.join(results_dir_path, "model_epoch_checkpoint"),
                                     global_step=i_epoch)
            tools.printf("Checkpoint saved")

        if tensorboard_meta: writer.flush()

        tools.printf("Epoch %d complete...\n%s ave_val_loss(se3): %f\ntime: %.2f\n" %
                     (i_epoch, train_losses_log_set.epoch_string(i_epoch), ave_val_loss, time.time() - start_time))

        train_data_gen.next_epoch()

    tools.printf("Final save...")
    train_losses_log_set.write_to_disk(results_dir_path)
    val_losses_log.write_to_disk(results_dir_path)
    tf_checkpoint_saver.save(sess, os.path.join(results_dir_path, "model_epoch_checkpoint"), global_step=i_epoch)
    tools.printf("Saved results to %s" % results_dir_path)

    sess.close()
