import data_roller as data
import config
import model
import tools
import tensorflow as tf
import numpy as np
import os
import pykitti
import transformations

dir_name = "trajectory_results"
kitti_seqs =["00", "01", "02", "08", "09"]
# kitti_seqs = ["04", "05", "06", "07", "10"]
# kitti_seqs = ["00", "01", "02", "04", "05", "06", "07", "08", "09", "10"]
# kitti_seqs = ["04"]
restore_model_file = "/home/cs4li/Dev/end_to_end_odometry/results/train_seq_20180712-10-52-15_16ts_golden/model_epoch_checkpoint-199"

save_ground_truth = True
config_class = config.SeqTrainLidarConfig
config.print_configs(config_class)
cfg = config_class()
cfg_si = config_class()

# Manipulate the configurations for evaluation
cfg.timesteps = 1
cfg.sequence_stride = 1
cfg.batch_size = 1
cfg.bidir_aug = False
cfg.use_init = False

cfg_si.timesteps = 1
cfg_si.sequence_stride = 1
cfg_si.batch_size = 1
cfg_si.bidir_aug = False
# cfg_si.use_init = what ever the original setting was

tools.printf("Building eval model....")
inputs, lstm_initial_state, initial_poses, imu_data, ekf_initial_state, ekf_initial_covariance, _, _ \
    = model.seq_model_inputs(cfg)
fc_outputs, fc_covar, se3_outputs, lstm_states, ekf_states, ekf_covar_states, _, _, _ = \
    model.build_seq_model(cfg, inputs, lstm_initial_state, initial_poses, imu_data, ekf_initial_state,
                          ekf_initial_covariance,
                          tf.constant(False, dtype=tf.bool),  # is training
                          False,  # get_activation
                          tf.constant(False, dtype=tf.bool),  # use initializer
                          cfg.use_ekf)  # use ekf

if cfg_si.use_init:
    tools.printf("Building eval model for initial LSTM states...")
    inputs_si, _, initial_poses_si, imu_data_si, _, _, _, _ = model.seq_model_inputs(cfg_si)
    _, _, _, _, _, _, feed_lstm_initial_states, feed_ekf_inital_states, feed_initial_covariance = \
        model.build_seq_model(cfg_si, inputs_si,
                              tf.constant(np.zeros([2, cfg_si.lstm_layers, cfg_si.batch_size, cfg_si.lstm_size]),
                                          dtype=tf.float32),
                              initial_poses_si,
                              imu_data_si,
                              np.zeros([cfg_si.batch_size, 17], dtype=np.float32),
                              0.01 * np.repeat(np.expand_dims(np.identity(17, dtype=np.float32), axis=0),
                                               repeats=cfg_si.batch_size, axis=0),
                              tf.constant(False, dtype=tf.bool),  # is training
                              False,
                              tf.constant(True, dtype=tf.bool),  # use initializer
                              cfg_si.use_ekf)

for kitti_seq in kitti_seqs:
    tools.printf("Loading eval data...")
    data_gen = data.StatefulRollerDataGen(cfg, config.dataset_path, [kitti_seq])
    if cfg_si.use_init:
        tools.printf("Loading eval data for initial LSTM states...")
        data_gen_si = data.StatefulRollerDataGen(cfg_si, config.dataset_path, [kitti_seq],
                                                 frames=[range(0, cfg_si.timesteps + 1)])

    results_dir_path = os.path.join(config.save_path, dir_name)
    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)

    # ==== Read Model Checkpoints =====
    variable_to_load = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "^(?!optimizer).*")
    tf_restore_saver = tf.train.Saver(variable_to_load)

    with tf.Session() as sess:
        tools.printf("Restoring model weights from %s..." % restore_model_file)
        tf_restore_saver.restore(sess, restore_model_file)

        total_batches = data_gen.total_batches()
        tools.printf("Start evaluation loop...")

        prediction = np.zeros([total_batches + 1, 7])
        ground_truth = np.zeros([total_batches + 1, 7])
        init_pose = np.expand_dims(data_gen.get_sequence_initial_pose(kitti_seq), axis=0)
        prediction[0, :] = init_pose
        ground_truth[0, :] = init_pose

        # if an initializer is used, get the current starting LSTM state from running network
        if cfg_si.use_init:
            tools.printf("Calculating initial LSTM state from initializer network...")
            _, _, batch_data_si, _, _, imu_meas_si = data_gen_si.next_batch()
            curr_lstm_states, curr_ekf_state, curr_ekf_cov_state = sess.run(
                    [feed_lstm_initial_states, feed_ekf_inital_states, feed_initial_covariance],
                    feed_dict={
                        inputs_si: batch_data_si,
                        initial_poses_si: init_pose,
                        imu_data_si: imu_meas_si
                    },
            )
        else:
            tools.printf("Using default initial states...")
            curr_lstm_states = np.zeros([2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size], dtype=np.float32)
            curr_ekf_state = np.zeros([cfg.batch_size, 17], dtype=np.float32)
            curr_ekf_cov_state = 0.01 * np.repeat(np.expand_dims(np.identity(17, dtype=np.float32), axis=0),
                                                  repeats=cfg.batch_size, axis=0)
        while data_gen.has_next_batch():
            j_batch = data_gen.curr_batch()

            # get inputs
            _, _, batch_data, fc_ground_truth, se3_ground_truth, imu_meas = data_gen.next_batch()

            # Run training session
            _curr_lstm_states, _se3_outputs, _fc_outputs = sess.run(
                    [lstm_states, se3_outputs, fc_outputs],
                    feed_dict={
                        inputs: batch_data,
                        lstm_initial_state: curr_lstm_states,
                        initial_poses: init_pose,
                        imu_data: imu_meas
                    },
            )
            curr_lstm_states = _curr_lstm_states
            init_pose = _se3_outputs[-1]

            prediction[j_batch + 1, :] = _se3_outputs[-1, -1]
            ground_truth[j_batch + 1, :] = se3_ground_truth[-1, -1]

            if j_batch % 100 == 0:
                tools.printf("Processed %.2f%%" % (data_gen.curr_batch() / data_gen.total_batches() * 100))

        # The trajectory is in the frame of the sensor, now put it back to cam0
        data_odom_kitti = pykitti.odometry(config.dataset_path, kitti_seq)
        if cfg.data_type == "cam" and cfg.input_channels == 3:
            T_cam2_cam0 = data_odom_kitti.calib.T_cam2_velo. \
                dot(np.linalg.inv(data_odom_kitti.calib.T_cam0_velo))
            T_cam0_xxx = np.linalg.inv(T_cam2_cam0)
        elif cfg.data_type == "lidar":
            T_cam0_xxx = data_odom_kitti.calib.T_cam0_velo
        else:
            T_cam0_xxx = np.eye(4)  # else cam0 transformation is just identity

        T_cam0_xxx_inv = np.linalg.inv(T_cam0_xxx)

        for i in range(0, prediction.shape[0]):
            pose_tf = transformations.quaternion_matrix(prediction[i, 3:7])
            pose_tf[0:3, 3] = prediction[i, 0:3]

            gt_tf = transformations.quaternion_matrix(ground_truth[i, 3:7])
            gt_tf[0:3, 3] = ground_truth[i, 0:3]

            pose_tf = np.dot(T_cam0_xxx, np.dot(pose_tf, T_cam0_xxx_inv))
            gt_tf = np.dot(T_cam0_xxx, np.dot(gt_tf, T_cam0_xxx_inv))

            prediction[i, 0:3] = pose_tf[0:3, 3]
            prediction[i, 3:7] = transformations.quaternion_from_matrix(pose_tf)

            ground_truth[i, 0:3] = gt_tf[0:3, 3]
            ground_truth[i, 3:7] = transformations.quaternion_from_matrix(gt_tf)

        # save the trajectories
        np.save(os.path.join(results_dir_path, "trajectory_" + kitti_seq), prediction)
        if save_ground_truth:
            np.save(os.path.join(results_dir_path, "ground_truth_" + kitti_seq), ground_truth)
