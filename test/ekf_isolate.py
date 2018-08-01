import ekf
import tensorflow as tf
import numpy as np
import config
import data_roller
import tools
import os
import model

kitti_seq = "06"
# frames = [range(0, 100)]
frames = [None]


class SeqTrainLidarConfig:
    timesteps = 1
    sequence_stride = 1
    batch_size = 1

    input_width = 1152
    input_height = 64
    input_channels = 2

    bidir_aug = False
    use_init = False

    data_type = "lidar"

    nes = 6


cfg = SeqTrainLidarConfig
initial_state_covar = 1
imu_covar = 0.1
fc_covar_val = 0.1

gyro_bias_diag = np.array([imu_covar] * 3, dtype=np.float32)
acc_bias_diag = np.array([imu_covar] * 3, dtype=np.float32)
gyro_covar_diag = np.array([imu_covar] * 3, dtype=np.float32)
acc_covar_diag = np.array([imu_covar] * 3, dtype=np.float32)

gyro_bias_covar = tf.diag((gyro_bias_diag) + 1e-4)
acc_bias_covar = tf.diag((acc_bias_diag) + 1e-4)
gyro_covar = tf.diag((gyro_covar_diag) + 1e-4)
acc_covar = tf.diag((acc_covar_diag) + 1e-4)

imu_data = tf.placeholder(tf.float32, shape=[cfg.timesteps, cfg.batch_size, 6])
fc_outputs = tf.placeholder(tf.float32, shape=[cfg.timesteps, cfg.batch_size, 12])
ekf_initial_state = tf.placeholder(tf.float32, name="ekf_init_state", shape=[cfg.batch_size, cfg.nes])
ekf_initial_covariance = tf.placeholder(tf.float32, name="ekf_init_covar", shape=[cfg.batch_size, cfg.nes, cfg.nes])
initial_poses = tf.placeholder(tf.float32, name="initial_poses", shape=[cfg.batch_size, 7])

stack1 = []
for i in range(fc_outputs.shape[0]):
    stack2 = []
    for j in range(fc_outputs.shape[1]):
        stack2.append(tf.diag(tf.square(fc_outputs[i, j, 6:])))
    stack1.append(tf.stack(stack2, axis=0))

nn_covar = tf.stack(stack1, axis=0)

ekf_out_states, ekf_out_covar, Kk = ekf.rotation_only_ekf(imu_data, fc_outputs[..., 0:6], nn_covar,
                                                          ekf_initial_state, ekf_initial_covariance,
                                                          gyro_bias_covar, gyro_covar)

rel_disp = tf.concat([fc_outputs[..., 0:3], ekf_out_states[1:, :, 0:3]], axis=-1)
rel_covar = tf.concat([tf.concat([nn_covar[:, :, 0:3, 0:3], tf.zeros(nn_covar[:, :, 0:3, 0:3].shape)], axis=-1),
                       tf.concat([tf.zeros(nn_covar[:, :, 0:3, 0:3].shape), ekf_out_covar[1:, :, 0:3, 0:3]], axis=-1)],
                      axis=-2)

se3_outputs = model.se3_layer(rel_disp, initial_poses)

data_gen = data_roller.StatefulRollerDataGen(cfg, config.dataset_path, [kitti_seq], frames=frames)

results_dir_path = os.path.join(config.save_path, "ekf_debug")
if not os.path.exists(results_dir_path):
    os.makedirs(results_dir_path)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    total_batches = data_gen.total_batches()
    tools.printf("Start evaluation loop...")

    prediction = np.zeros([total_batches + 1, 7])
    ground_truths = np.zeros([total_batches + 1, 7])
    ekf_states = np.zeros([total_batches + 1, cfg.nes])
    fc_ground_truths = np.zeros([total_batches + 1, 6])
    imu_measurements = np.zeros([total_batches + 1, 6])
    init_pose = np.expand_dims(data_gen.get_sequence_initial_pose(kitti_seq), axis=0)
    prediction[0, :] = init_pose
    ground_truths[0, :] = init_pose
    fc_ground_truths[0, :] = np.zeros([6])
    imu_measurements[0, :] = np.zeros([6])

    curr_ekf_state = np.zeros([cfg.batch_size, cfg.nes], dtype=np.float32)
    # curr_ekf_state[:, [3]] = 12  # !!! initial state
    ekf_states[0, :] = curr_ekf_state
    curr_ekf_cov_state = initial_state_covar * np.repeat(np.expand_dims(np.identity(cfg.nes, dtype=np.float32), axis=0),
                                                         repeats=cfg.batch_size, axis=0)

    while data_gen.has_next_batch():
        j_batch = data_gen.curr_batch()

        _, _, batch_data, fc_ground_truth, se3_ground_truth, imu_meas = data_gen.next_batch()
        fc_covar = np.reshape(np.array([fc_covar_val] * 6, dtype=np.float32), [1, 1, 6])
        fc_outputs_input = np.concatenate([fc_ground_truth, fc_covar, ], axis=2)

        _se3_outputs, _curr_ekf_states, _curr_ekf_cov_state, _Kk = sess.run(
                [se3_outputs, ekf_out_states, ekf_out_covar, Kk],
                feed_dict={
                    fc_outputs: fc_outputs_input,
                    initial_poses: init_pose,
                    imu_data: imu_meas,
                    ekf_initial_state: curr_ekf_state,
                    ekf_initial_covariance: curr_ekf_cov_state,
                },
        )
        init_pose = _se3_outputs[-1]
        curr_ekf_state = _curr_ekf_states[-1]
        curr_ekf_cov_state = _curr_ekf_cov_state[-1]

        print(j_batch)
        print(_Kk)
        # print(fc_outputs_input[-1, -1, 3:6])
        print(curr_ekf_state)
        # print(curr_ekf_cov_state)

        prediction[j_batch + 1, :] = _se3_outputs[-1, -1]
        ground_truths[j_batch + 1, :] = se3_ground_truth[-1, -1]
        ekf_states[j_batch + 1, :] = curr_ekf_state
        fc_ground_truths[j_batch + 1, :] = fc_ground_truth
        imu_measurements[j_batch + 1, :] = imu_meas

        if j_batch % 100 == 0:
            tools.printf("Processed %.2f%%" % (data_gen.curr_batch() / data_gen.total_batches() * 100))

    # save the trajectories
    np.save(os.path.join(results_dir_path, "%s_trajectory" % kitti_seq), prediction)
    np.save(os.path.join(results_dir_path, "%s_ground_truth" % kitti_seq), ground_truths)
    np.save(os.path.join(results_dir_path, "%s_fc_ground_truth" % kitti_seq), fc_ground_truths)
    np.save(os.path.join(results_dir_path, "%s_ekf_states" % kitti_seq), ekf_states)
    np.save(os.path.join(results_dir_path, "%s_imu_measurements" % kitti_seq), imu_measurements)
