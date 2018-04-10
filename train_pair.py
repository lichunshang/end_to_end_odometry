import data
import config
import tools

cfg = config.PairTrainConfigs
config.print_configs(cfg)

tools.printf("Loading training data...")
train_data_gen = data.StatefulDataGen(cfg, "/home/cs4li/Dev/KITTI/dataset/",
                                      ["00", "02", "05", "08", "09"])
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
inputs, is_training, fc_outputs = model.build_pair_model(cfg)
_, fc_labels = model.model_labels(cfg)

tools.printf("Building losses...")
with tf.device("/gpu:0"):
    with tf.variable_scope("Losses"):
        fc_losses, fc_xyz_losses, fc_ypr_losses, x_ave_losses, y_ave_losses, z_ave_losses = losses.pair_train_fc_losses(
            fc_outputs, fc_labels, cfg.k)

tools.printf("Building optimizer...")
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
                is_training: False
            }
        )

        fc_losses_history.append(_fc_losses)

    return fc_losses_history, sum(fc_losses_history) / len(fc_losses_history)


# =================== SAVING/LOADING DATA ========================
results_dir_path = tools.create_results_dir("train_pair")

cnn_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "^cnn_layer.*")
tf_init_saver = tf.train.Saver(cnn_variables)
init_model_file = "/home/cs4li/Dev/end_to_end_visual_odometry/results/" \
                  "flownet_weights/flownet_s_weights"

tf_saver = tf.train.Saver(max_to_keep=5)
restore_model_file = None

# =================== TRAINING ========================
with tf.Session() as sess:
    if init_model_file:
        tools.printf("Init model weights from %s..." % init_model_file)
        sess.run(tf.global_variables_initializer())
        tf_init_saver.restore(sess, init_model_file)
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
    fc_losses_history = np.zeros([cfg.num_epochs, total_batches])
    fc_val_losses_history = np.zeros([cfg.num_epochs, val_data_gen.total_batches()])
    best_val_loss = 9999999999

    tools.printf("Start training loop...")
    for i_epoch in range(cfg.num_epochs):
        tools.printf("Training Epoch: %d ..." % i_epoch)

        train_data_gen.next_epoch()

        start_time = time.time()
        _fc_losses = 0

        epoch_fc_losses_history = []

        while train_data_gen.has_next_batch():
            init_poses, reset_state, batch_data, \
            fc_ground_truth, _ = train_data_gen.next_batch_random()

            _fc_outputs, _fc_losses, _fc_trainer, _fc_xyz_losses, _fc_ypr_losses, \
            _x_ave_losses, _y_ave_losses, _z_ave_losses, = sess.run(
                [fc_outputs, fc_losses, fc_trainer, fc_xyz_losses, fc_ypr_losses,
                 x_ave_losses, y_ave_losses, z_ave_losses, ],
                feed_dict={
                    inputs: batch_data,
                    fc_labels: fc_ground_truth,
                    fc_lr: 0.001,
                    is_training: True
                }
            )

            epoch_fc_losses_history.append(_fc_losses)

            # print stats
            tools.printf("batch %d/%d: fc_loss: %.3f, fc_xyz: %.5f, fc_ypr: %.5f, x: %.5f, y: %.5f, z: %.5f" % (
                train_data_gen.curr_batch(), train_data_gen.total_batches(),
                _fc_losses, _fc_xyz_losses, _fc_ypr_losses,
                _x_ave_losses, _y_ave_losses, _z_ave_losses))

        ave_fc_loss = sum(epoch_fc_losses_history) / total_batches
        fc_losses_history[i_epoch, :] = epoch_fc_losses_history
        epoch_fc_val_losses, ave_val_loss = calc_val_loss(sess)
        fc_val_losses_history[i_epoch, :] = epoch_fc_val_losses

        if ave_val_loss < best_val_loss:
            best_val_loss = ave_val_loss
            tf_saver.save(sess, os.path.join(results_dir_path, "model_best_val_checkpoint"), global_step=i_epoch)
            np.save(os.path.join(results_dir_path, "fc_losses_history"), fc_losses_history)
            np.save(os.path.join(results_dir_path, "fc_val_losses_history"), fc_val_losses_history)
            tools.printf("Best val loss, model saved.")
        elif i_epoch % 10 == 0:
            tf_saver.save(sess, os.path.join(results_dir_path, "model_epoch_checkpoint"), global_step=i_epoch)
            np.save(os.path.join(results_dir_path, "fc_losses_history"), fc_losses_history)
            np.save(os.path.join(results_dir_path, "fc_val_losses_history"), fc_val_losses_history)
            tools.printf("Checkpoint saved")

        tools.printf("Epoch %d, ave_fc_loss: %.3f, ave_val_loss: %f, time: %.2f" %
                     (i_epoch, ave_fc_loss, ave_val_loss, time.time() - start_time))
        tools.printf()

    np.save(os.path.join(results_dir_path, "fc_losses_history"), fc_losses_history)
    np.save(os.path.join(results_dir_path, "fc_val_losses_history"), fc_val_losses_history)
    tf_saver.save(sess, os.path.join(results_dir_path, "model_epoch_checkpoint"), global_step=i_epoch)
    tools.printf("Saved results to %s" % results_dir_path)
