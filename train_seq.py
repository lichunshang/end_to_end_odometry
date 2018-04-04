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
lr_set = 0.001
start_epoch = 0
alpha_schedule = {0: 0.99,  # epoch: alpha
                  10: 0.9,
                  20: 0.5,
                  30: 0.1,
                  40: 0.025}

cfg = config.SeqTrainConfigs

# =================== MODEL + LOSSES + Optimizer ========================
inputs, lstm_initial_state, initial_poses, is_training, fc_outputs, se3_outputs, lstm_states = model.build_seq_training_model()
se3_labels, fc_labels = model.model_labels(cfg)

tools.printf("Building losses...")
with tf.device("/gpu:0"):
    with tf.variable_scope("Losses"):
        se3_losses = losses.se3_losses(se3_outputs, se3_labels, cfg.k)
        fc_losses = losses.fc_losses(fc_outputs, fc_labels)
        alpha = tf.placeholder(tf.float32, name="alpha", shape=[])  # between 0 and 1, larger favors fc loss
        total_losses = (1 - alpha) * se3_losses + alpha * fc_losses

tools.printf("Building optimizer...")
with tf.variable_scope("Optimizer"):
    # dynamic learning rates
    lr = tf.placeholder(tf.float32, name="se3_lr", shape=[])
    with tf.device("/gpu:0"):
        trainer = tf.train.AdamOptimizer(learning_rate=lr).minimize(total_losses, colocate_gradients_with_ops=True)

# ================ LOADING DATASET ===================

tools.printf("Loading training data...")
train_data_gen = data.StatefulDataGen(cfg, "/home/cs4li/Dev/KITTI/dataset/",
                                      ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"])
# train_data_gen = data.StatefulDataGen(cfg, "/home/cs4li/Dev/KITTI/dataset/", ["01"], frames=[range(0, 100)])
tools.printf("Loading validation data...")
val_data_gen = data.StatefulDataGen(cfg, "/home/cs4li/Dev/KITTI/dataset/", ["10"], frames=[None])


# for evaluating validation loss
def calc_val_loss(sess):
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
        se3_losses_history.append(_se3_losses)

    return se3_losses_history, sum(se3_losses_history) / len(se3_losses_history)


# =================== SAVING/LOADING DATA ========================
results_dir_path = tools.create_results_dir("train_seq")
tf_saver = tf.train.Saver()
restore_model_file = None

# just for restoring pre trained cnn weights
cnn_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "^cnn_layer.*")
cnn_init_tf_saver = tf.train.Saver(cnn_variables)
cnn_init_model_file = "/home/cs4li/Dev/end_to_end_visual_odometry/results/" \
                      "train_pair_20180402-12-21-24_seq_00_to_05_randomized_dropout(0.9, 0.8, 0.7)/" \
                      "model_best_val_checkpoint"

# =================== TRAINING ========================
config = tf.ConfigProto(allow_soft_placement=True)
with tf.Session(config=config) as sess:
    if cnn_init_model_file:
        tools.printf("Taking initialization weights from %s..." % cnn_init_model_file)
        sess.run(tf.global_variables_initializer())
        cnn_init_tf_saver.restore(sess, cnn_init_model_file)
    elif restore_model_file:
        tools.printf("Restoring model weights from %s..." % restore_model_file)
        tf_saver.restore(sess, restore_model_file)
    else:
        tools.printf("Initializing variables...")
        sess.run(tf.global_variables_initializer())

    # Visualization
    writer = tf.summary.FileWriter('graph_viz/')
    writer.add_graph(tf.get_default_graph())

    total_batches = train_data_gen.total_batches()
    total_losses_history = np.zeros([cfg.num_epochs, total_batches])
    val_losses_history = np.zeros([cfg.num_epochs, val_data_gen.total_batches()])
    best_val_loss = 9999999999
    alpha_set = -1

    tools.printf("Start training loop...")
    tools.printf("lr: %f" % lr_set)
    tools.printf("start_epoch: %f" % start_epoch)
    tools.printf("alpha_schedule: %s" % alpha_schedule)
    for i_epoch in range(start_epoch, cfg.num_epochs):
        tools.printf("Training Epoch: %d ..." % i_epoch)

        curr_lstm_states = np.zeros([2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size])
        start_time = time.time()
        _total_losses = 0
        epoch_total_losses_history = []

        if i_epoch in alpha_schedule.keys():
            alpha_set = alpha_schedule[i_epoch]
            tools.printf("alpha set to %f" % alpha_set)

        train_data_gen.next_epoch()

        while train_data_gen.has_next_batch():
            init_poses, reset_state, batch_data, \
            fc_ground_truth, se3_ground_truth = train_data_gen.next_batch()

            curr_lstm_states = data.reset_select_lstm_state(curr_lstm_states, reset_state)

            _total_losses, trainer, _curr_lstm_states = sess.run(
                [total_losses, trainer, lstm_states, ],
                feed_dict={
                    inputs: batch_data,
                    se3_labels: se3_ground_truth,
                    fc_labels: fc_ground_truth,
                    lstm_initial_state: curr_lstm_states,
                    initial_poses: init_poses,
                    lr: lr_set,
                    alpha: alpha_set,
                    is_training: True
                }
            )

            epoch_total_losses_history.append(_total_losses)

            curr_lstm_states = _curr_lstm_states

            # print stats
            tools.printf("batch %d/%d: total_loss: %.3f" % (
                train_data_gen.curr_batch(), train_data_gen.total_batches(), _total_losses))

        ave_total_loss = sum(epoch_total_losses_history) / total_batches

        total_losses_history[i_epoch, :] = epoch_total_losses_history

        epoch_val_losses, ave_val_loss = calc_val_loss(sess)
        val_losses_history[i_epoch, :] = epoch_val_losses

        if ave_val_loss < best_val_loss:
            best_val_loss = ave_val_loss
            tf_saver.save(sess, os.path.join(results_dir_path, "model_best_val_checkpoint"))
            tools.printf("Best val loss, model saved.")
        elif i_epoch % 5 == 0:
            tf_saver.save(sess, os.path.join(results_dir_path, "model_epoch_checkpoint"))
            tools.printf("Checkpoint saved")

        tools.printf("Epoch %d, ave_total_loss: %.3f, ave_val_loss(se3): %f, time: %.2f" %
                     (i_epoch, ave_total_loss, ave_val_loss, time.time() - start_time))
        tools.printf()

    np.save(os.path.join(results_dir_path, "total_losses_history"), total_losses_history)
    np.save(os.path.join(results_dir_path, "val_losses_history_se3"), val_losses_history)
    tf_saver.save(sess, os.path.join(results_dir_path, "model_epoch_checkpoint"))
    tools.printf("Saved results to %s" % results_dir_path)
