import data

print("Loading training data...")
# train_data_gen = data.StatefulDataGen("/home/lichunshang/Dev/KITTI/dataset/",
#                                       ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"])
train_data_gen = data.StatefulDataGen("/home/lichunshang/Dev/KITTI/dataset/", ["01"], frames=range(0, 100))
print("Loading validation data...")
val_data_gen = data.StatefulDataGen("/home/lichunshang/Dev/KITTI/dataset/", ["10"], frames=range(0, 100))

import os
import model
import losses
from config import *
import tensorflow as tf
import numpy as np
import time
import tools

# =================== MODEL + LOSSES + Optimizer ========================
inputs, lstm_initial_state, initial_poses, fc_outputs, se3_outputs, lstm_states = model.build_seq_training_model()
se3_labels, fc_labels = model.model_labels()

print("Building losses...")
with tf.device("/gpu:0"):
    with tf.variable_scope("Losses"):
        se3_losses = losses.se3_losses(se3_outputs, se3_labels, k)
        fc_losses = losses.fc_losses(fc_outputs, fc_labels)

print("Building optimizer...")
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
    curr_lstm_states = np.zeros([2, lstm_layers, batch_size, lstm_size])

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
            }
        )

        curr_lstm_states = _curr_lstm_states
        se3_losses_history.append(_se3_losses)

    return se3_losses_history, sum(se3_losses_history) / len(se3_losses_history)


# =================== SAVING/LOADING DATA ========================
results_dir_path = tools.create_results_dir("train_seq")
tf_saver = tf.train.Saver()
restore_model_file = None

# =================== TRAINING ========================
with tf.Session() as sess:
    if restore_model_file:
        print("Restoring model weights from %s..." % restore_model_file)
        tf_saver.restore(sess, restore_model_file)
    else:
        print("Initializing variables...")
        sess.run(tf.global_variables_initializer())

    # Visualization
    writer = tf.summary.FileWriter('graph_viz/')
    writer.add_graph(tf.get_default_graph())

    total_batches = train_data_gen.total_batches()
    se3_losses_history = np.zeros([num_epochs, total_batches])
    fc_losses_history = np.zeros([num_epochs, total_batches])
    se3_val_losses_history = np.zeros([num_epochs, val_data_gen.total_batches()])
    best_val_loss = 9999999999

    print("Start training loop...")
    for i_epoch in range(num_epochs):
        print("Training Epoch: %d ..." % i_epoch)

        train_data_gen.next_epoch()
        curr_lstm_states = np.zeros([2, lstm_layers, batch_size, lstm_size])

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
                    fc_lr: 0.001,
                }
            )

            _se3_outputs, _se3_losses, _se3_trainer, _curr_lstm_states = sess.run(
                [se3_outputs, se3_losses, se3_trainer, lstm_states, ],
                feed_dict={
                    inputs: batch_data,
                    se3_labels: se3_ground_truth,
                    lstm_initial_state: curr_lstm_states,
                    initial_poses: init_poses,
                    se3_lr: 0.001,
                }
            )

            epoch_se3_losses_history.append(_se3_losses)
            epoch_fc_losses_history.append(_fc_losses)

            curr_lstm_states = _curr_lstm_states

            # print stats
            print("batch %d/%d: se3_loss: %.3f, fc_loss: %.3f" % (
                train_data_gen.curr_batch(), train_data_gen.total_batches(), _se3_losses, _fc_losses))

        ave_se3_loss = sum(epoch_se3_losses_history) / total_batches
        ave_fc_loss = sum(epoch_fc_losses_history) / total_batches

        se3_losses_history[i_epoch, :] = epoch_se3_losses_history
        fc_losses_history[i_epoch, :] = epoch_fc_losses_history

        epoch_se3_val_losses, ave_val_loss = calc_val_loss(sess)
        se3_val_losses_history[i_epoch, :] = epoch_se3_val_losses

        if ave_val_loss < best_val_loss:
            tf_saved_path = tf_saver.save(sess, os.path.join(results_dir_path))
            print("Best val loss, model saved.")

        print("Epoch %d, ave_se3_loss: %.3f, ave_fc_loss: %.3f, ave_val_loss: %f, time: %.2f" %
              (i_epoch, ave_se3_loss, ave_fc_loss, ave_val_loss, time.time() - start_time))
        print()

    print("Saving results to %s" % results_dir_path)
    np.save(os.path.join(results_dir_path, "se3_losses_history"), se3_losses_history)
    np.save(os.path.join(results_dir_path, "fc_losses_history"), fc_losses_history)
    np.save(os.path.join(results_dir_path, "se3_val_losses_history"), se3_val_losses_history)