import data
import config
import tools
import os
import model
import losses
import tensorflow as tf
import numpy as np
import time

# =================== CONFIGURATIONS ========================
cfg = config.SeqTrainConfigs
config.print_configs(cfg)

lr_set = 0.001
start_epoch = 0
# alpha_schedule = {0: 0.99,  # epoch: alpha
#                   20: 0.9,
#                   40: 0.5,
#                   60: 0.1,
#                   80: 0.025}
alpha_schedule = {0: 0.99,
                  50: 0.25}

tensorboard_meta = False

# =================== MODEL + LOSSES + Optimizer ========================
inputs, lstm_initial_state, initial_poses, is_training, fc_outputs, se3_outputs, lstm_states = model.build_seq_training_model()
se3_labels, fc_labels = model.model_labels(cfg)

tools.printf("Building losses...")
with tf.device("/gpu:0"):
    with tf.variable_scope("Losses"):
        se3_losses, se3_xyz_losses, se3_quat_losses = losses.se3_losses(se3_outputs, se3_labels, cfg.k_se3)
        fc_losses, fc_xyz_losses, fc_ypr_losses = losses.pair_train_fc_losses(fc_outputs, fc_labels, cfg.k_fc)
        alpha = tf.placeholder(tf.float32, name="alpha", shape=[])  # between 0 and 1, larger favors fc loss
        total_losses = (1 - alpha) * se3_losses + alpha * fc_losses

tools.printf("Building optimizer...")
with tf.variable_scope("Optimizer"):
    # dynamic learning rates
    lr = tf.placeholder(tf.float32, name="se3_lr", shape=[])
    trainer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_losses, colocate_gradients_with_ops=False)

# ================ LOADING DATASET ===================

tools.printf("Loading training data...")
train_data_gen = data.StatefulDataGen(cfg, "/home/cs4li/Dev/KITTI/dataset/",
                                      ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"])
# train_data_gen = data.StatefulDataGen(cfg, "/home/cs4li/Dev/KITTI/dataset/", ["03"], frames=[None])
tools.printf("Loading validation data...")
val_data_gen = data.StatefulDataGen(cfg, "/home/cs4li/Dev/KITTI/dataset/", ["10"], frames=[None])


# for evaluating validation loss
def calc_val_loss(sess, i_epoch, losses_log):
    curr_lstm_states = np.zeros([2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size])

    se3_losses_history = []
    val_data_gen.next_epoch()

    while val_data_gen.has_next_batch():
        init_poses, reset_state, batch_data, _, se3_ground_truth = val_data_gen.next_batch()

        curr_lstm_states = data.reset_select_lstm_state(curr_lstm_states, reset_state)

        _se3_losses, _curr_lstm_states = sess.run(
            [se3_losses, lstm_states, ],
            feed_dict={
                inputs: batch_data,
                lstm_initial_state: curr_lstm_states,
                initial_poses: init_poses,
                se3_labels: se3_ground_truth,
                is_training: False
            }
        )

        curr_lstm_states = _curr_lstm_states
        losses_log.log(i_epoch, val_data_gen.curr_batch() - 1, _se3_losses)

    return losses_log


# =================== SAVING/LOADING DATA ========================
results_dir_path = tools.create_results_dir("train_seq")
tf_checkpoint_saver = tf.train.Saver(max_to_keep=3)
tf_best_saver = tf.train.Saver(max_to_keep=2)

tf_restore_saver = tf.train.Saver()
restore_model_file = None
# restore_model_file = "/home/cs4li/Dev/end_to_end_visual_odometry/results/" \
#                      "train_seq_20180406-17-59-50_seq_all_cnn_init_cosine_dist/" \
#                      "model_epoch_checkpoint-49"

# just for restoring pre trained cnn weights
cnn_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "^cnn_layer.*")
cnn_init_tf_saver = tf.train.Saver(cnn_variables)
# cnn_init_model_file = None
cnn_init_model_file = "/home/cs4li/Dev/end_to_end_visual_odometry/results/" \
                      "train_pair_20180402-12-21-24_seq_00_to_05_randomized_dropout(0.9, 0.8, 0.7)/" \
                      "model_best_val_checkpoint"

# =================== TRAINING ========================
# config = tf.ConfigProto(allow_soft_placement=True)
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
    writer = tf.summary.FileWriter('graph_viz/')
    writer.add_graph(tf.get_default_graph())
    writer.flush()

    # Set up for training
    total_batches = train_data_gen.total_batches()
    train_losses_log_set = tools.LossesSet(["total", "fc", "se3", "fc_xyz", "fc_ypr", "se3_xyz", "se3_quat"],
                                           cfg.num_epochs, train_data_gen.total_batches())
    val_losses_log = tools.Losses("se3_val", cfg.num_epochs, val_data_gen.total_batches())

    tools.printf("Start training loop...")
    tools.printf("lr: %f" % lr_set)
    tools.printf("start_epoch: %f" % start_epoch)
    tools.printf("alpha_schedule: %s" % alpha_schedule)

    best_val_loss = 9999999999
    alpha_set = -1
    i_epoch = 0
    for i_epoch in range(start_epoch, cfg.num_epochs):
        tools.printf("Training Epoch: %d ..." % i_epoch)

        curr_lstm_states = np.zeros([2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size])
        start_time = time.time()

        if i_epoch in alpha_schedule.keys():
            alpha_set = alpha_schedule[i_epoch]
            tools.printf("alpha set to %f" % alpha_set)

        train_data_gen.next_epoch()

        while train_data_gen.has_next_batch():
            j_batch = train_data_gen.curr_batch()
            # get inputs
            init_poses, reset_state, batch_data, \
            fc_ground_truth, se3_ground_truth = train_data_gen.next_batch()
            curr_lstm_states = data.reset_select_lstm_state(curr_lstm_states, reset_state)

            # Run training session
            _trainer, _curr_lstm_states, _total_losses, _fc_losses, _se3_losses, \
            _fc_xyz_losses, _fc_ypr_losses, _se3_xyz_losses, _se3_quat_losses = sess.run(
                [trainer, lstm_states, total_losses, fc_losses, se3_losses,
                 fc_xyz_losses, fc_ypr_losses, se3_xyz_losses, se3_quat_losses],
                feed_dict={
                    inputs: batch_data,
                    se3_labels: se3_ground_truth,
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

            # for tensorboard
            if tensorboard_meta: writer.add_run_metadata(run_metadata, 'epochid=%d_batchid=%d' % (i_epoch, j_batch))

            train_losses_log_set.log({
                "total": _total_losses, "fc": _fc_losses, "se3": _se3_losses,
                "fc_xyz": _fc_xyz_losses, "fc_ypr": _fc_ypr_losses,
                "se3_xyz": _se3_xyz_losses, "se3_quat": _se3_quat_losses
            }, i_epoch, j_batch)

            # print stats
            tools.printf("batch %d/%d: %s" % (
                train_data_gen.curr_batch(), train_data_gen.total_batches(),
                train_losses_log_set.batch_string(i_epoch, j_batch)))

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

    tools.printf("Final save...")
    train_losses_log_set.write_to_disk(results_dir_path)
    val_losses_log.write_to_disk(results_dir_path)
    tf_checkpoint_saver.save(sess, os.path.join(results_dir_path, "model_epoch_checkpoint"), global_step=i_epoch)
    tools.printf("Saved results to %s" % results_dir_path)

    sess.close()
