import data_roller as data
import config
import model
import tools
import tensorflow as tf
import numpy as np
import os

dir_name = "trajectory_results"
kitti_seqs = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

save_ground_truth = True
config_class = config.SeqTrainLidarConfig
config.print_configs(config_class)
cfg = config_class()
cfg_state_init = config_class()

# Manipulate the configurations for evaluation
cfg.timesteps = 1
cfg.sequence_stride = 1
cfg.batch_size = 1
cfg.bidir_aug = False
cfg.use_init = False
cfg_state_init.batch_size = 1

tools.printf("Building eval model....")
inputs, lstm_initial_state, initial_poses, _, _ = model.seq_model_inputs(cfg)
fc_outputs, se3_outputs, lstm_states, _ = \
    model.build_seq_model(cfg, inputs, lstm_initial_state, initial_poses, tf.constant(False, dtype=tf.bool),
                          tf.constant(False, dtype=tf.bool))

tools.printf("Building eval model for initial LSTM states...")
inputs_state_init, _, _, _, _ = model.seq_model_inputs(cfg)
_, _, _, feed_init_states = \
    model.build_seq_model(cfg_state_init, inputs_state_init,
                          tf.constant(np.zeros([2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size]), dtype=tf.float32),
                          tf.constant(np.array([[0., 0., 0., 1., 0., 0., 0.]]), dtype=tf.float32),
                          tf.constant(False, dtype=tf.bool),
                          tf.constant(False, dtype=tf.bool))

for kitti_seq in kitti_seqs:
    tools.printf("Loading eval data...")
    data_gen = data.StatefulRollerDataGen(cfg, config.dataset_path, [kitti_seq])
    tools.printf("Loading eval data for initial LSTM states...")
    data_gen_state_init = data.StatefulRollerDataGen(cfg_state_init, config.dataset_path, [kitti_seq],
                                                     frames=[range(0, cfg_state_init.timesteps)])

    results_dir_path = os.path.join(config.save_path, dir_name)
    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)

    # ==== Read Model Checkpoints =====
    restore_model_file = "/home/cs4li/Dev/end_to_end_visual_odometry/results/train_seq_20180509-00-04-46_2channel128batchsize2gpu/model_epoch_checkpoint-125"

    variable_to_load = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "^(cnn_layer|rnn_layer|fc_layer).*")
    tf_restore_saver = tf.train.Saver(variable_to_load)

    with tf.Session() as sess:
        tools.printf("Restoring model weights from %s..." % restore_model_file)
        tf_restore_saver.restore(sess, restore_model_file)

        total_batches = data_gen.total_batches()
        tools.printf("Start evaluation loop...")

        prediction = np.zeros([total_batches, 7])
        ground_truth = np.zeros([total_batches, 7])
        init_pose = np.array([[0., 0., 0., 1., 0., 0., 0.]], dtype=np.float32)

        # if an initializer is used, get the current starting LSTM state from running network
        if cfg_state_init.use_init:
            _, _, batch_data_init_state, _, _ = data_gen_state_init.next_batch()
            curr_lstm_states = sess.run(
                    feed_init_states,
                    feed_dict={
                        inputs: batch_data_init_state,
                    },
            )
        else:
            curr_lstm_states = np.zeros([2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size])

        while data_gen.has_next_batch():
            j_batch = data_gen.curr_batch()

            # get inputs
            _, _, batch_data, fc_ground_truth, se3_ground_truth = data_gen.next_batch()

            # Run training session
            _curr_lstm_states, _se3_outputs, _fc_outputs = sess.run(
                    [lstm_states, se3_outputs, fc_outputs],
                    feed_dict={
                        inputs: batch_data,
                        lstm_initial_state: curr_lstm_states,
                        initial_poses: init_pose,
                    },
            )
            curr_lstm_states = _curr_lstm_states
            init_pose = _se3_outputs[-1]

            prediction[j_batch, :] = _se3_outputs[-1, -1]
            ground_truth[j_batch, :] = se3_ground_truth[-1, -1]

            if j_batch % 100 == 0:
                tools.printf("Processed %.2f%%" % (data_gen.curr_batch() / data_gen.total_batches() * 100))

        np.save(os.path.join(results_dir_path, "trajectory_" + kitti_seq), prediction)
        if save_ground_truth:
            np.save(os.path.join(results_dir_path, "ground_truth_" + kitti_seq), ground_truth)
