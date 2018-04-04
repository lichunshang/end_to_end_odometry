import data
import config
import tools

cfg = config.SeqTrainConfigs

tools.printf("Loading training data...")
train_data_gen = data.StatefulDataGen(cfg, "/home/cs4li/Dev/KITTI/dataset/",
                                      ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"])
# train_data_gen = data.StatefulDataGen(cfg, "/home/cs4li/Dev/KITTI/dataset/", ["01"], frames=[range(0, 100)])
tools.printf("Loading validation data...")
val_data_gen = data.StatefulDataGen(cfg, "/home/cs4li/Dev/KITTI/dataset/", ["10"], frames=[None])

import os
import model
import losses
import tensorflow as tf
import numpy as np
import time
import tools

# =================== MODEL + LOSSES + Optimizer ========================
inputs, lstm_initial_state, initial_poses, is_training, fc_outputs, se3_outputs, lstm_states = model.build_seq_training_model()
se3_labels, fc_labels = model.model_labels(cfg)

tools.printf("Building losses...")
with tf.device("/gpu:0"):
    with tf.variable_scope("Losses"):
        se3_losses = losses.se3_losses(se3_outputs, se3_labels, cfg.k)
        fc_losses = losses.fc_losses(fc_outputs, fc_labels)

tools.printf("Building optimizer...")
with tf.variable_scope("Optimizer"):
    # dynamic learning rates
    se3_lr = tf.placeholder(tf.float32, name="se3_lr", shape=[])
    fc_lr = tf.placeholder(tf.float32, name="fc_lr", shape=[])
    with tf.device("/gpu:0"):
        se3_trainer = tf.train.AdamOptimizer(learning_rate=se3_lr).minimize(se3_losses)
    with tf.device("/gpu:0"):
        fc_trainer = tf.train.AdamOptimizer(learning_rate=fc_lr).minimize(fc_losses)


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
with tf.Session() as sess:
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
    se3_losses_history = np.zeros([cfg.num_epochs, total_batches])
    fc_losses_history = np.zeros([cfg.num_epochs, total_batches])
    se3_val_losses_history = np.zeros([cfg.num_epochs, val_data_gen.total_batches()])
    fc_lr_set = 0.001
    se3_lr_set = 0.001
    best_val_loss = 9999999999

    tools.printf("Start training loop...")
    for i_epoch in range(cfg.num_epochs):
        tools.printf("Training Epoch: %d ..." % i_epoch)

        train_data_gen.next_epoch()
        curr_lstm_states = np.zeros([2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size])

        start_time = time.time()
        _se3_losses = 0
        _fc_losses = 0

        epoch_se3_losses_history = []
        epoch_fc_losses_history = []

        while train_data_gen.has_next_batch():
            init_poses, reset_state, batch_data, \
            fc_ground_truth, se3_ground_truth = train_data_gen.next_batch()

            curr_lstm_states = data.reset_select_lstm_state(curr_lstm_states, reset_state)

            _fc_outputs, _fc_losses, _fc_trainer = sess.run(
                [fc_outputs, fc_losses, fc_trainer],
                feed_dict={
                    inputs: batch_data,
                    fc_labels: fc_ground_truth,
                    lstm_initial_state: curr_lstm_states,
                    fc_lr: fc_lr_set,
                    is_training: True
                }
            )

            _se3_outputs, _se3_losses, _se3_trainer, _curr_lstm_states = sess.run(
                [se3_outputs, se3_losses, se3_trainer, lstm_states, ],
                feed_dict={
                    inputs: batch_data,
                    se3_labels: se3_ground_truth,
                    lstm_initial_state: curr_lstm_states,
                    initial_poses: init_poses,
                    se3_lr: se3_lr_set,
                    is_training: True
                }
            )

            epoch_se3_losses_history.append(_se3_losses)
            epoch_fc_losses_history.append(_fc_losses)

            curr_lstm_states = _curr_lstm_states

            # print stats
            tools.printf("batch %d/%d: se3_loss: %.3f, fc_loss: %.3f" % (
                train_data_gen.curr_batch(), train_data_gen.total_batches(), _se3_losses, _fc_losses))

        ave_se3_loss = sum(epoch_se3_losses_history) / total_batches
        ave_fc_loss = sum(epoch_fc_losses_history) / total_batches

        se3_losses_history[i_epoch, :] = epoch_se3_losses_history
        fc_losses_history[i_epoch, :] = epoch_fc_losses_history

        epoch_se3_val_losses, ave_val_loss = calc_val_loss(sess)
        se3_val_losses_history[i_epoch, :] = epoch_se3_val_losses

        if ave_val_loss < best_val_loss:
            best_val_loss = ave_val_loss
            tf_saver.save(sess, os.path.join(results_dir_path, "model_best_val_checkpoint"))
            tools.printf("Best val loss, model saved.")
        elif i_epoch % 5 == 0:
            tf_saver.save(sess, os.path.join(results_dir_path, "model_epoch_checkpoint"))
            tools.printf("Checkpoint saved")

        tools.printf("Epoch %d, ave_se3_loss: %.3f, ave_fc_loss: %.3f, ave_val_loss: %f, time: %.2f" %
                     (i_epoch, ave_se3_loss, ave_fc_loss, ave_val_loss, time.time() - start_time))
        tools.printf()

    np.save(os.path.join(results_dir_path, "se3_losses_history"), se3_losses_history)
    np.save(os.path.join(results_dir_path, "fc_losses_history"), fc_losses_history)
    np.save(os.path.join(results_dir_path, "se3_val_losses_history"), se3_val_losses_history)
    tf_saver.save(sess, os.path.join(results_dir_path, "model_epoch_checkpoint"))
    tools.printf("Saved results to %s" % results_dir_path)
