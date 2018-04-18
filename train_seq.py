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
# cfg = config.SeqTrainConfigs
cfg = config.SeqTrainLidarConfig
val_cfg = config.SeqTrainConfigsSmallStepsValidation
config.print_configs(cfg)

lr_set = 0.0001
lr_schedule = {
    0:   0.0001,
    40:  0.00008,
    70:  0.00005,
    80:  0.000002,
    100: 0.000001
}
# lr_schedule = {
#     0:   0.00001,
#     40:  0.00001,
#     70:  0.00001,
#     80:  0.000002,
#     100: 0.000001
# }
start_epoch = 0
# alpha_schedule = {0: 0.99,  # epoch: alpha
#                   20: 0.9,
#                   40: 0.5,
#                   60: 0.1,
#                   80: 0.025}
alpha_schedule = {0: 1,
                  10: 0.99,
                  15: 0.9,
                  25: 0.8,
                  35: 0.5}

tensorboard_meta = False

# =================== MODEL + LOSSES + Optimizer ========================
tools.printf("Building losses...")
with tf.device("/cpu:0"):
    alpha = tf.placeholder(tf.float32, name="alpha", shape=[])  # between 0 and 1, larger favors fc loss

with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device='/job:localhost/replica:0/task:0/device:GPU:0',
                                              worker_device='/job:localhost/replica:0/task:0/device:GPU:0')):
    inputs, lstm_initial_state, initial_poses, is_training, fc_outputs, se3_outputs, lstm_states = model.build_seq_model(
        cfg, True)
    se3_labels, fc_labels = simple_model.model_labels(cfg)

    with tf.variable_scope("Losses"):
        se3_losses, se3_xyz_losses, se3_quat_losses = losses.se3_losses(se3_outputs, se3_labels, cfg.k_se3)
        fc_losses, fc_xyz_losses, fc_ypr_losses, \
        x_loss, y_loss, z_loss = losses.pair_train_fc_losses(fc_outputs, fc_labels, cfg.k_fc)
        total_losses = (1 - alpha) * se3_losses + alpha * fc_losses

tools.printf("Building optimizer...")
with tf.variable_scope("Optimizer"):
    with tf.device("/gpu:0"):
        # dynamic learning rates
        lr = tf.placeholder(tf.float32, name="se3_lr", shape=[])
        trainer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_losses, colocate_gradients_with_ops=True)

# with tf.device(tf.train.replica_device_setter(ps_tasks=1, ps_device='/job:localhost/replica:0/task:0/device:CPU:0', worker_device='/job:localhost/replica:0/task:0/device:GPU:1')):
#     val_inputs, val_lstm_init, val_initial_poses, val_is_training, val_fc_outputs, val_se3_outputs, val_lstm_states = simple_model.build_seq_model(
#         val_cfg, True)
#     val_se3_labels, val_fc_labels = simple_model.model_labels(val_cfg)
#
#     with tf.variable_scope("Val_Losses"):
#         se3_losses_val, _, _ = losses.se3_losses(val_se3_outputs, val_se3_labels, val_cfg.k_se3)
#         fc_losses_val, _, _, _, _, _ = losses.pair_train_fc_losses(val_fc_outputs, val_fc_labels, val_cfg.k_fc)
#         total_losses_val = (1 - alpha) * se3_losses_val + alpha * fc_losses_val

# =================== SAVING/LOADING DATA ========================
results_dir_path = tools.create_results_dir("train_seq")
tools.log_file_content(results_dir_path, os.path.realpath(__file__))

tf_checkpoint_saver = tf.train.Saver(max_to_keep=3)
tf_best_saver = tf.train.Saver(max_to_keep=2)

tf_restore_saver = tf.train.Saver()
restore_model_file = None
# restore_model_file = "/home/cs4li/Dev/end_to_end_visual_odometry/results/train_seq_20180413-18-29-33/model_epoch_checkpoint-99"

# just for restoring pre trained cnn weights
cnn_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "^cnn_layer.*")
cnn_init_tf_saver = tf.train.Saver(cnn_variables)
cnn_init_model_file = None
# cnn_init_model_file = "/home/cs4li/Dev/end_to_end_visual_odometry/results/train_seq_20180414-01-33-38_simplemodel1lstmseq0f2f/model_epoch_checkpoint-199"
# cnn_init_model_file = "/home/cs4li/Dev/end_to_end_visual_odometry/results/" \
#                       "flownet_weights/flownet_s_weights"

# =================== TRAINING ========================
# config = tf.ConfigProto(allow_soft_placement=True)

sequence_id = tf.placeholder(dtype=tf.uint8, shape=[])

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
tf.summary.scalar("alpha", alpha)
tf.summary.scalar("lr", lr)
tf.summary.scalar("sequence_id", sequence_id)

train_merged_summary_op = tf.summary.merge_all()

activations = tf.get_collection(tf.GraphKeys.ACTIVATIONS)
initial_layer = tf.summary.image("1st layer activations", tf.expand_dims(activations[0][:, 0, :, :], -1))
final_layer = tf.summary.image("Last layer activations", tf.expand_dims(activations[1][:, 0, :, :], -1))

image_summary_op = tf.summary.merge([initial_layer, final_layer])

# val_loss_sum = tf.summary.scalar("training_loss_val", total_losses_val)
# val_fc_sum = tf.summary.scalar("fc_losses_val", fc_losses_val)
# val_se3_sum = tf.summary.scalar("se3_losses_val", se3_losses_val)
#
# val_merged_summary_op = tf.summary.merge([val_loss_sum, val_fc_sum, val_se3_sum])

val_loss_sum = tf.summary.scalar("training_loss_val", total_losses)
val_fc_sum = tf.summary.scalar("fc_losses_val", fc_losses)
val_se3_sum = tf.summary.scalar("se3_losses_val", se3_losses)
val_z_sum = tf.summary.scalar("z_loss_val", z_loss)

val_merged_summary_op = tf.summary.merge([val_loss_sum, val_fc_sum, val_se3_sum, val_z_sum])

# ================ LOADING DATASET ===================

tools.printf("Loading training data...")
train_sequences = ["01", "09"]
train_data_gen = data.StatefulRollerDataGen(cfg, config.dataset_path, train_sequences,
                                            frames=None)
tools.printf("Loading validation data...")
validation_sequences = ["07"]
val_data_gen = data.StatefulRollerDataGen(cfg, config.dataset_path, validation_sequences,
                                          frames=[range(500), ])

# ============== For Validation =============
def calc_val_loss(sess, writer, i_epoch, alpha_set, run_options, run_metadata):
    curr_lstm_states = np.zeros([2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size])

    val_data_gen.next_epoch(randomize=False)

    val_se3_losses_log = np.zeros([val_data_gen.total_batches()])

    while val_data_gen.has_next_batch():
        j_batch = val_data_gen.curr_batch()

        # never resets state, there will only be one sequence in validation
        _, _, batch_data, fc_ground_truth, se3_ground_truth = val_data_gen.next_batch()

        init_poses = se3_ground_truth[0, :, :]
        _curr_lstm_states, _summary, _total_losses, _se3_losses = sess.run(
            [lstm_states, val_merged_summary_op, total_losses, se3_losses],
            feed_dict={
                inputs: batch_data,
                se3_labels: se3_ground_truth[1:, :, :],
                fc_labels: fc_ground_truth,
                lstm_initial_state: curr_lstm_states,
                initial_poses: init_poses,
                alpha: alpha_set,
                is_training: False
            },
            options=run_options,
            run_metadata=run_metadata
        )

        curr_lstm_states = np.stack(_curr_lstm_states, 0)
        writer.add_summary(_summary, i_epoch * val_data_gen.total_batches() + j_batch)
        val_se3_losses_log[j_batch] = _se3_losses

    return np.average(val_se3_losses_log)


# ============ Training Session ============
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

    writer = tf.summary.FileWriter(os.path.join(results_dir_path, 'graph_viz'))
    writer.add_graph(tf.get_default_graph())
    writer.flush()

    # Set up for training
    total_batches = train_data_gen.total_batches()

    tools.printf("Start training loop...")
    tools.printf("lr: %f" % lr_set)
    tools.printf("start_epoch: %f" % start_epoch)
    tools.printf("alpha_schedule: %s" % alpha_schedule)

    best_val_loss = 9999999999
    alpha_set = 0.8
    i_epoch = 0

    # for evaluating validation loss
    curr_val_loss = 9999999999999
    val_curr_lstm_states = np.zeros([2, val_cfg.lstm_layers, val_cfg.batch_size, val_cfg.lstm_size])
    n_states = len(train_sequences)
    lstm_states_dic = {}
    curr_lstm_states = np.zeros([2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size], dtype=np.float32)
    for seq in train_sequences:
        lstm_states_dic[seq] = np.zeros([train_data_gen.batch_counts[seq], 2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size], dtype=np.float32)

    for i_epoch in range(start_epoch, cfg.num_epochs):
        tools.printf("Training Epoch: %d ..." % i_epoch)
        start_time = time.time()

        if i_epoch in alpha_schedule.keys():
            alpha_set = alpha_schedule[i_epoch]
            tools.printf("alpha set to %f" % alpha_set)

        if i_epoch in lr_schedule.keys():
            lr_set = lr_schedule[i_epoch]
            tools.printf("learning rate set to %f" % lr_set)

        init_poses = np.zeros([cfg.batch_size, 7], dtype=np.float32)
        init_poses[:, 3] = np.ones([cfg.batch_size], dtype=np.float32)

        val_init_poses = np.zeros([cfg.batch_size, 7], dtype=np.float32)
        val_init_poses[:, 3] = np.ones([cfg.batch_size], dtype=np.float32)

        while train_data_gen.has_next_batch():
            j_batch = train_data_gen.curr_batch()

            # reset validation epoch if required
            # if not val_data_gen.has_next_batch():
            #     if val_cfg.bidir_aug:
            #         mid = val_cfg.batch_size / 2
            #         mid = int(mid)
            #         val_curr_lstm_states[:, :, 1:mid, :] = val_curr_lstm_states[:, :, 0:(mid - 1), :]
            #         val_curr_lstm_states[:, :, mid + 1:, :] = val_curr_lstm_states[:, :, mid:-1, :]
            #         val_curr_lstm_states[:, :, mid, :] = np.zeros(val_curr_lstm_states[:, :, mid, :].shape, dtype=np.float32)
            #     else:
            #         val_curr_lstm_states[:, :, 1:, :] = val_curr_lstm_states[:, :, 0:-1, :]
            #     val_curr_lstm_states[:, :, 0, :] = np.zeros(val_curr_lstm_states[:, :, 0, :].shape, dtype=np.float32)
            #
            #     val_data_gen.next_epoch()

            # get inputs
            batch_id, curr_seq, batch_data, fc_ground_truth, se3_ground_truth = train_data_gen.next_batch()

            # never need to reset, only one sequence in validation data
            # _, val_batch_data, val_fc_ground_truth, val_se3_ground_truth = val_data_gen.next_batch()

            data.get_init_lstm_state(lstm_states_dic, curr_lstm_states, curr_seq, batch_id, cfg.bidir_aug)

            # shift se3 ground truth to be relative to the first pose
            init_poses = se3_ground_truth[0, :, :]
            # val_init_poses = val_se3_ground_truth[0, :, :]

            # Run training session
            _, _curr_lstm_states, _se3_outputs, _train_summary, _train_image_summary, _total_losses = sess.run(
                [trainer, lstm_states, se3_outputs, train_merged_summary_op, image_summary_op, total_losses],
                feed_dict={
                    inputs: batch_data,
                    se3_labels: se3_ground_truth[1:, :, :],
                    fc_labels: fc_ground_truth,
                    lstm_initial_state: curr_lstm_states,
                    initial_poses: init_poses,
                    lr: lr_set,
                    alpha: alpha_set,
                    is_training: True,
                    sequence_id: int(curr_seq)
                },
                options=run_options,
                run_metadata=run_metadata
            )
            # _, _curr_lstm_states, _se3_outputs, _train_summary, _total_losses, _val_curr_lstm_states, _val_summary, _val_loss = sess.run(
            #     [trainer, lstm_states, se3_outputs, train_merged_summary_op, total_losses, val_lstm_states, val_merged_summary_op, total_losses_val],
            #     feed_dict={
            #         inputs: batch_data,
            #         se3_labels: se3_ground_truth[1:,:,:],
            #         fc_labels: fc_ground_truth,
            #         lstm_initial_state: curr_lstm_states,
            #         initial_poses: init_poses,
            #         lr: lr_set,
            #         alpha: alpha_set,
            #         is_training: True,
            #         val_is_training: False,
            #         val_inputs: val_batch_data,
            #         val_se3_labels: val_se3_ground_truth[1:, :, :],
            #         val_fc_labels: val_fc_ground_truth,
            #         val_lstm_init: val_curr_lstm_states,
            #         val_initial_poses: init_poses,
            #     },
            #     options=run_options,
            #     run_metadata=run_metadata
            # )

            data.update_lstm_state(lstm_states_dic, np.stack(_curr_lstm_states, 0), curr_seq, batch_id)

            # val_curr_lstm_states = np.stack(_val_curr_lstm_states, 0)
            # curr_val_loss = _val_loss

            # for tensorboard
            if tensorboard_meta:
                writer.add_run_metadata(run_metadata, 'epochid=%d_batchid=%d' % (i_epoch, j_batch))

            writer.add_summary(_train_summary, i_epoch * train_data_gen.total_batches() + j_batch)

            if j_batch % 100 == 0:
                writer.add_summary(_train_image_summary, i_epoch * train_data_gen.total_batches() + j_batch)

            # writer.add_summary(_val_summary, i_epoch * train_data_gen.total_batches() + j_batch)

            # print stats
            # tools.printf("batch %d/%d: Loss:%.7f  Validation Loss:%.7f" % (
            #     train_data_gen.curr_batch(), train_data_gen.total_batches(),
            #     _total_losses, _val_loss))
            tools.printf("batch %d/%d: Loss:%.7f" % (
                j_batch + 1, train_data_gen.total_batches(),
                _total_losses))

        tools.printf("Evaluating validation loss...")

        curr_val_loss = calc_val_loss(sess, writer, i_epoch, alpha_set, run_options, run_metadata)

        # check for best results
        if curr_val_loss < best_val_loss:
            tools.printf("Saving best result...")
            best_val_loss = curr_val_loss
            tf_best_saver.save(sess, os.path.join(results_dir_path, "best_val", "model_best_val_checkpoint"),
                               global_step=i_epoch)
            tools.printf("Best val loss, model saved.")
        if i_epoch % 5 == 0:
            tools.printf("Saving checkpoint...")
            tf_checkpoint_saver.save(sess, os.path.join(results_dir_path, "model_epoch_checkpoint"),
                                     global_step=i_epoch)
            tools.printf("Checkpoint saved")

        if tensorboard_meta: writer.flush()

        tools.printf("ave_val_loss(se3): %f, time: %f\n" % (curr_val_loss, time.time() - start_time))

        train_data_gen.next_epoch()

    tools.printf("Final save...")
    tf_checkpoint_saver.save(sess, os.path.join(results_dir_path, "model_epoch_checkpoint"), global_step=i_epoch)
    tools.printf("Saved results to %s" % results_dir_path)

    sess.close()
