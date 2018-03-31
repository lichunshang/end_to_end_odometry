import data
import config

cfg = config.PairTrainConfigs

print("Loading training data...")
# train_data_gen = data.StatefulDataGen(cfg, "/home/lichunshang/Dev/KITTI/dataset/",
#                                       ["00"])
train_data_gen = data.StatefulDataGen(cfg, "/home/lichunshang/Dev/KITTI/dataset/", ["01"], frames=range(0, 100))
print("Loading validation data...")
val_data_gen = data.StatefulDataGen(cfg, "/home/lichunshang/Dev/KITTI/dataset/", ["10"], frames=range(0, 100))

import os
import model
import losses
import tensorflow as tf
import numpy as np
import time
import tools

# =================== MODEL + LOSSES + Optimizer ========================
inputs, fc_outputs = model.build_pair_training_model()
_, fc_labels = model.model_labels(cfg)

print("Building losses...")
with tf.device("/gpu:0"):
    with tf.variable_scope("Losses"):
        fc_losses = losses.pair_train_fc_losses(fc_outputs, fc_labels, cfg.k)

print("Building optimizer...")
with tf.variable_scope("Optimizer"):
    # dynamic learning rates
    fc_lr = tf.placeholder(tf.float32, name="fc_lr", shape=[])
    with tf.device("/gpu:0"):
        fc_trainer = tf.train.AdamOptimizer(learning_rate=fc_lr).minimize(fc_losses)


# for evaluating validation loss
def calc_val_loss(sess):
    fc_losses_history = []
    val_data_gen.next_epoch()

    while val_data_gen.has_next_batch():
        init_poses, reset_state, batch_data, fc_ground_truth, _ = val_data_gen.next_batch()

        _fc_losses = sess.run(
            fc_losses,
            feed_dict={
                inputs: batch_data,
                fc_labels: fc_ground_truth,
            }
        )

        fc_losses_history.append(_fc_losses)

    return fc_losses_history, sum(fc_losses_history) / len(fc_losses_history)


# =================== SAVING/LOADING DATA ========================
results_dir_path = tools.create_results_dir("train_pair")
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
    fc_losses_history = np.zeros([cfg.num_epochs, total_batches])
    fc_val_losses_history = np.zeros([cfg.num_epochs, val_data_gen.total_batches()])
    best_val_loss = 9999999999

    print("Start training loop...")
    for i_epoch in range(cfg.num_epochs):
        print("Training Epoch: %d ..." % i_epoch)

        train_data_gen.next_epoch()

        start_time = time.time()
        _fc_losses = 0

        epoch_fc_losses_history = []

        while train_data_gen.has_next_batch():
            init_poses, reset_state, batch_data, \
            fc_ground_truth, _ = train_data_gen.next_batch()

            _fc_outputs, _fc_losses, _fc_trainer = sess.run(
                [fc_outputs, fc_losses, fc_trainer],
                feed_dict={
                    inputs: batch_data,
                    fc_labels: fc_ground_truth,
                    fc_lr: 0.001,
                }
            )

            epoch_fc_losses_history.append(_fc_losses)

            # print stats
            print("batch %d/%d: fc_loss: %.3f" % (
                train_data_gen.curr_batch(), train_data_gen.total_batches(), _fc_losses))

        ave_fc_loss = sum(epoch_fc_losses_history) / total_batches
        fc_losses_history[i_epoch, :] = epoch_fc_losses_history
        epoch_fc_val_losses, ave_val_loss = calc_val_loss(sess)
        fc_val_losses_history[i_epoch, :] = epoch_fc_val_losses

        if ave_val_loss < best_val_loss:
            tf_saved_path = tf_saver.save(sess, os.path.join(results_dir_path, "model_checkpoint"))
            print("Best val loss, model saved.")

        print("Epoch %d, ave_fc_loss: %.3f, ave_val_loss: %f, time: %.2f" %
              (i_epoch, ave_fc_loss, ave_val_loss, time.time() - start_time))
        print()

    np.save(os.path.join(results_dir_path, "fc_losses_history"), fc_losses_history)
    np.save(os.path.join(results_dir_path, "fc_val_losses_history"), fc_val_losses_history)
    print("Saved results to %s" % results_dir_path)
