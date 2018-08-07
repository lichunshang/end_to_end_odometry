import tensorflow as tf
from tensorflow.python.ops.distributions.util import fill_triangular
import se3
import tools
import config
import numpy as np
import native_lstm
import ekf

tfe = tf.contrib.eager


# CNN Block
# is_training to control whether to apply dropout
def cnn_model(inputs, is_training, get_activations=False):
    with tf.variable_scope("cnn_model"):
        conv_1 = tf.contrib.layers.conv2d(inputs, num_outputs=64, kernel_size=(7, 7,),
                                          stride=(2, 2), padding="same", scope="conv_1", data_format="NCHW",
                                          weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                          biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                          activation_fn=tf.nn.leaky_relu)

        if get_activations:
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, conv_1)

        dropout_conv_1 = tf.contrib.layers.dropout(conv_1, keep_prob=1.0, is_training=is_training,
                                                   scope="dropout_conv_1")
        conv_2 = tf.contrib.layers.conv2d(dropout_conv_1, num_outputs=128, kernel_size=(5, 5,),
                                          stride=(2, 2), padding="same", scope="conv_2", data_format="NCHW",
                                          weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                          biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                          activation_fn=tf.nn.leaky_relu)
        dropout_conv_2 = tf.contrib.layers.dropout(conv_2, keep_prob=1.0, is_training=is_training,
                                                   scope="dropout_conv_2")

        conv_3 = tf.contrib.layers.conv2d(dropout_conv_2, num_outputs=256, kernel_size=(5, 5,),
                                          stride=(2, 2), padding="same", scope="conv_3", data_format="NCHW",
                                          weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                          biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                          activation_fn=tf.nn.leaky_relu)
        dropout_conv_3 = tf.contrib.layers.dropout(conv_3, keep_prob=1, is_training=is_training,
                                                   scope="dropout_conv_3")
        conv_3_1 = tf.contrib.layers.conv2d(conv_3, num_outputs=256, kernel_size=(3, 3,),
                                            stride=(1, 1), padding="same", scope="conv_3_1", data_format="NCHW",
                                            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                            biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                            activation_fn=tf.nn.leaky_relu)
        dropout_conv_3_1 = tf.contrib.layers.dropout(conv_3_1, keep_prob=1, is_training=is_training,
                                                     scope="dropout_conv_3_1")

        conv_4 = tf.contrib.layers.conv2d(conv_3_1, num_outputs=512, kernel_size=(3, 3,),
                                          stride=(2, 2), padding="same", scope="conv_4", data_format="NCHW",
                                          weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                          biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                          activation_fn=tf.nn.leaky_relu)
        dropout_conv_4 = tf.contrib.layers.dropout(conv_4, keep_prob=0.9, is_training=is_training,
                                                   scope="dropout_conv_4")
        conv_4_1 = tf.contrib.layers.conv2d(dropout_conv_4, num_outputs=512, kernel_size=(3, 3,),
                                            stride=(1, 1), padding="same", scope="conv_4_1", data_format="NCHW",
                                            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                            biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                            activation_fn=tf.nn.leaky_relu)
        dropout_conv_4_1 = tf.contrib.layers.dropout(conv_4_1, keep_prob=0.9, is_training=is_training,
                                                     scope="dropout_conv_4_1")

        conv_5 = tf.contrib.layers.conv2d(dropout_conv_4_1, num_outputs=512, kernel_size=(3, 3,),
                                          stride=(2, 2), padding="same", scope="conv_5", data_format="NCHW",
                                          weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                          biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                          activation_fn=tf.nn.leaky_relu)
        dropout_conv_5 = tf.contrib.layers.dropout(conv_5, keep_prob=0.8, is_training=is_training,
                                                   scope="dropout_conv_5")
        conv_5_1 = tf.contrib.layers.conv2d(dropout_conv_5, num_outputs=512, kernel_size=(3, 3,),
                                            stride=(1, 1), padding="same", scope="conv_5_1", data_format="NCHW",
                                            weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                            biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                            activation_fn=tf.nn.leaky_relu)
        dropout_conv_5_1 = tf.contrib.layers.dropout(conv_5_1, keep_prob=0.8, is_training=is_training,
                                                     scope="dropout_conv_5_1")

        conv_6 = tf.contrib.layers.conv2d(dropout_conv_5_1, num_outputs=1024, kernel_size=(3, 3,),
                                          stride=(2, 2), padding="same", scope="conv_6", data_format="NCHW",
                                          activation_fn=None,
                                          weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                          biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004))
        dropout_conv_6 = tf.contrib.layers.dropout(conv_6, keep_prob=0.7, is_training=is_training,
                                                   scope="dropout_conv_6")
        return dropout_conv_6


def cnn_model_lidar(inputs, is_training, get_activations=False):
    with tf.variable_scope("cnn_model"):
        # The first kernel is a 1d convolution
        conv_1 = tf.contrib.layers.conv2d(inputs, num_outputs=64, kernel_size=(1, 7,),
                                          stride=(1, 1), padding="same", scope="conv_1", data_format="NCHW")

        if get_activations:
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, conv_1)

        conv_2 = tf.contrib.layers.conv2d(conv_1, num_outputs=128, kernel_size=(1, 5,),
                                          stride=(1, 2), padding="same", scope="conv_2", data_format="NCHW")

        conv_3 = tf.contrib.layers.conv2d(conv_2, num_outputs=240, kernel_size=(3, 5,),
                                          stride=(2, 2), padding="same", scope="conv_3", data_format="NCHW")

        # conv_3_1 = tf.contrib.layers.conv2d(dropout_conv_3, num_outputs=240, kernel_size=(3, 3,),
        #                                     stride=(1, 1), padding="same", scope="conv_3_1", data_format="NCHW")
        # dropout_conv_3_1 = tf.contrib.layers.dropout(conv_3_1, keep_prob=1, is_training=is_training,
        #                                              scope="dropout_conv_3_1")

        conv_4 = tf.contrib.layers.conv2d(conv_3, num_outputs=450, kernel_size=(3, 3,),
                                          stride=(2, 2), padding="same", scope="conv_4", data_format="NCHW")
        dropout_conv_4 = tf.contrib.layers.dropout(conv_4, keep_prob=0.9, is_training=is_training,
                                                   scope="dropout_conv_4")
        # conv_4_1 = tf.contrib.layers.conv2d(dropout_conv_4, num_outputs=450, kernel_size=(3, 3,),
        #                                     stride=(1, 1), padding="same", scope="conv_4_1", data_format="NCHW")
        # dropout_conv_4_1 = tf.contrib.layers.dropout(conv_4_1, keep_prob=0.9, is_training=is_training,
        #                                              scope="dropout_conv_4_1")

        conv_5 = tf.contrib.layers.conv2d(dropout_conv_4, num_outputs=450, kernel_size=(3, 3,),
                                          stride=(2, 2), padding="same", scope="conv_5", data_format="NCHW")
        dropout_conv_5 = tf.contrib.layers.dropout(conv_5, keep_prob=0.8, is_training=is_training,
                                                   scope="dropout_conv_5")
        # conv_5_1 = tf.contrib.layers.conv2d(dropout_conv_5, num_outputs=450, kernel_size=(3, 3,),
        #                                     stride=(1, 1), padding="same", scope="conv_5_1", data_format="NCHW")
        # dropout_conv_5_1 = tf.contrib.layers.dropout(conv_5_1, keep_prob=0.8, is_training=is_training,
        #                                              scope="dropout_conv_5_1")

        conv_6 = tf.contrib.layers.conv2d(dropout_conv_5, num_outputs=600, kernel_size=(3, 3,),
                                          stride=(1, 2), padding="same", scope="conv_6", data_format="NCHW",
                                          activation_fn=None)

        if get_activations:
            tf.add_to_collection(tf.GraphKeys.ACTIVATIONS, conv_6)

        dropout_conv_6 = tf.contrib.layers.dropout(conv_6, keep_prob=0.7, is_training=is_training,
                                                   scope="dropout_conv_6")
        return dropout_conv_6


def fc_model(inputs):
    with tf.variable_scope("pair_train_fc_model", reuse=tf.AUTO_REUSE):
        fc_128 = tf.contrib.layers.fully_connected(inputs, 128, scope="fc_128", activation_fn=tf.nn.relu,
                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005))
        fc_12 = tf.contrib.layers.fully_connected(fc_128, 12, scope="fc_12", activation_fn=None)
        return fc_12


def pair_train_fc_layer(inputs):
    with tf.variable_scope("pair_train_fc_model", reuse=tf.AUTO_REUSE):
        fc_128 = tf.contrib.layers.fully_connected(inputs, 128, scope="fc_128", activation_fn=tf.nn.relu,
                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005))
        fc_12 = tf.contrib.layers.fully_connected(fc_128, 12, scope="fc_12", activation_fn=None)
        return fc_12


def pair_train_fc_layer_1024(inputs):
    with tf.variable_scope("pair_train_fc_model", reuse=tf.AUTO_REUSE):
        fc_128 = tf.contrib.layers.fully_connected(inputs, 1024, scope="fc_128", activation_fn=tf.nn.relu)
        fc_6 = tf.contrib.layers.fully_connected(fc_128, 6, scope="fc_6", activation_fn=None)
        return fc_6


def cnn_over_timesteps(inputs, cnn_model, is_training, get_activations):
    with tf.variable_scope("cnn_over_timesteps"):
        unstacked_inputs = tf.unstack(inputs, axis=0)

        outputs = []

        for i in range(len(unstacked_inputs) - 1):
            # stack images along channels
            image_stacked = tf.concat((unstacked_inputs[i], unstacked_inputs[i + 1]), axis=1)
            outputs.append(cnn_model(image_stacked, is_training, get_activations))

        return tf.stack(outputs, axis=0)


def se3_comp_over_timesteps(inputs, initial_pose):
    with tf.variable_scope("se3_comp_over_timesteps"):
        # position + orientation in quat

        poses = []
        pose = initial_pose
        fc_ypr_poses = tf.unstack(inputs[:, 0:6], axis=0)  # take the x, y, z, y, p, r
        for d_ypr_pose in fc_ypr_poses:
            pose = se3.se3_comp(pose, d_ypr_pose)
            poses.append(pose)
        return tf.stack(poses)


def cnn_layer(inputs, cnn_model, is_training, get_activations):
    with tf.variable_scope("cnn_layer", reuse=tf.AUTO_REUSE):
        outputs = cnn_over_timesteps(inputs, cnn_model, is_training, get_activations)

    outputs = tf.reshape(outputs,
                         [outputs.shape[0], outputs.shape[1], outputs.shape[2] * outputs.shape[3] * outputs.shape[4]])

    return outputs


def rnn_layer(cfg, inputs, initial_state):
    with tf.variable_scope("rnn_layer", reuse=tf.AUTO_REUSE):
        initial_state = tuple(tf.unstack(initial_state))

        # need to break up LSTMs to get states in the middle
        mid_offset = cfg.sequence_stride
        if mid_offset < inputs.shape[0]:
            lstm1 = native_lstm.CudnnLSTM(cfg.lstm_layers, cfg.lstm_size, name='rnn_layer',
                                          variable_namespace="rnn_layer")
            lstm1.build(inputs[0:mid_offset, :, :].shape)
            mid_output, output_state = lstm1(inputs[0:mid_offset, :, :], initial_state=initial_state, training=True)

            lstm2 = native_lstm.CudnnLSTM(cfg.lstm_layers, cfg.lstm_size, name='rnn_layer',
                                          variable_namespace="rnn_layer")
            lstm2.build(inputs[mid_offset:, :, :].shape)
            end_output, _ = lstm2(inputs[mid_offset:, :, :], initial_state=output_state, training=True)

            outputs = tf.concat((mid_output, end_output), axis=0)
        else:
            lstm = tf.contrib.cudnn_rnn.CudnnLSTM(cfg.lstm_layers, cfg.lstm_size, name='rnn_layer')
            outputs, output_state = lstm(inputs, initial_state=initial_state, training=True)

        return outputs, output_state


def fc_layer(inputs, fc_model_fn=fc_model):
    with tf.variable_scope("fc_layer", reuse=tf.AUTO_REUSE):
        fc_outputs = tools.static_map_fn(fc_model_fn, inputs, axis=0)

    return fc_outputs


def se3_layer(inputs, initial_poses):
    with tf.variable_scope("se3_layer", reuse=tf.AUTO_REUSE):
        unstacked_inputs = tf.unstack(inputs, axis=1)
        unstacked_initial_poses = tf.unstack(initial_poses, axis=0)

        outputs = []

        for fc_timesteps, initial_pose in zip(unstacked_inputs, unstacked_initial_poses):
            outputs.append(se3_comp_over_timesteps(fc_timesteps, initial_pose))

        return tf.stack(outputs, axis=1)


def initializer_layer(inputs, cfg):
    with tf.variable_scope("initializer_layer", reuse=tf.AUTO_REUSE):
        print("Building Initializer Network")
        if cfg.init_length > inputs.shape[0]:
            raise ValueError("Invalid initializer size")
        init_list = tf.unstack(inputs[:cfg.init_length, ...], axis=0)
        init_feed = tf.concat(init_list, axis=1)
        initializer = tf.contrib.layers.fully_connected(init_feed, cfg.lstm_layers * 2 * cfg.lstm_size, scope="fc",
                                                        activation_fn=tf.nn.tanh)

        # Doing this manually to make sure data from different batches isn't mixed
        listed = tf.unstack(initializer, axis=0)
        states = []
        for elem in listed:
            states.append(tf.reshape(elem, [2, cfg.lstm_layers, cfg.lstm_size]))

        lstm_network_state = tf.stack(states, axis=2)

        # Another initializer for initial ekf state and covariance

        ekf_initializer = tf.contrib.layers.fully_connected(init_feed, 17 + 153, scope="fc_ekf", activation_fn=None)

        ekf_init_state = ekf_initializer[:, 0:17]

        sqrt_flat_L = ekf_initializer[:, 17:]

        L = fill_triangular(tf.square(sqrt_flat_L))

        ekf_init_covar = tf.matmul(L, L)

        return lstm_network_state, ekf_init_state, ekf_init_covar


def seq_model_inputs(cfg):
    # All time major
    inputs = tf.placeholder(tf.float32, name="inputs",
                            shape=[cfg.timesteps + 1, cfg.batch_size, cfg.input_channels,
                                   cfg.input_height, cfg.input_width])

    imu_data = tf.placeholder(tf.float32, name="imu_inputs", shape=[cfg.timesteps, cfg.batch_size, 6])

    # init LSTM states, 2 (cell + hidden states), 2 layers, batch size, and 1024 state size
    lstm_initial_state = tf.placeholder(tf.float32, name="lstm_init_state",
                                        shape=[2, cfg.lstm_layers, cfg.batch_size,
                                               cfg.lstm_size])

    # init EKF state. batch size * 17
    ekf_initial_state = tf.placeholder(tf.float32, name="ekf_init_state", shape=[cfg.batch_size, 17])
    # init EKF covariance. batch size * 17 * 17
    ekf_initial_covariance = tf.placeholder(tf.float32, name="ekf_init_covar", shape=[cfg.batch_size, 17, 17])

    # init poses, initial position for each example in the batch
    initial_poses = tf.placeholder(tf.float32, name="initial_poses", shape=[cfg.batch_size, 7])

    # is training
    is_training = tf.placeholder(tf.bool, name="is_training", shape=[])

    # switch between previous states and state initializer
    use_initializer = tf.placeholder(tf.bool, name="use_initializer", shape=[])

    if cfg.use_ekf:
        with tf.variable_scope("imu_noise_params", reuse=tf.AUTO_REUSE):
            gyro_bias_diag = tf.get_variable(name="gyro_bias_sqrt", shape=[3],
                                             initializer=tf.constant_initializer([cfg.init_gyro_bias_covar] * 3,
                                                                                 dtype=tf.float32),
                                             dtype=tf.float32, trainable=cfg.train_noise_covariance)

            acc_bias_diag = tf.get_variable(name="acc_bias_sqrt", shape=[3],
                                            initializer=tf.constant_initializer([cfg.init_acc_bias_covar] * 3,
                                                                                dtype=tf.float32),
                                            dtype=tf.float32, trainable=cfg.train_noise_covariance)

            gyro_covar_diag = tf.get_variable(name="gyro_sqrt", shape=[3],
                                              initializer=tf.constant_initializer([cfg.init_gyro_covar] * 3,
                                                                                  dtype=tf.float32),
                                              dtype=tf.float32, trainable=cfg.train_noise_covariance)

            acc_covar_diag = tf.get_variable(name="acc_sqrt", shape=[3],
                                             initializer=tf.constant_initializer([cfg.init_acc_covar] * 3,
                                                                                 dtype=tf.float32),
                                             dtype=tf.float32, trainable=cfg.train_noise_covariance)

    return inputs, lstm_initial_state, initial_poses, imu_data, ekf_initial_state, ekf_initial_covariance, is_training, use_initializer


def build_seq_model(cfg, inputs, lstm_initial_state, initial_poses, imu_data, ekf_initial_state, ekf_initial_covariance,
                    is_training, get_activations=False, use_initializer=False, use_ekf=False, fc_labels=None):
    print("Building CNN...")
    cnn_outputs = cnn_layer(inputs, cnn_model_lidar, is_training, get_activations)

    def f1():
        return initializer_layer(cnn_outputs, cfg)

    def f2():
        return lstm_initial_state, ekf_initial_state, ekf_initial_covariance

    with tf.name_scope("use_initializer_cond"):
        feed_init_states, feed_ekf_init_state, feed_ekf_init_covar = tf.cond(use_initializer, true_fn=f1, false_fn=f2)

    print("Building RNN...")
    lstm_outputs, lstm_states = rnn_layer(cfg, cnn_outputs, feed_init_states)

    print("Building FC...")
    # if we want to train ekf with ground truth, by passing all previous layers
    if cfg.train_ekf_with_fcgt:
        fc_outputs = fc_labels
    else:
        fc_outputs = fc_layer(lstm_outputs, fc_model)

    # if we want to fix covariances in fc_outputs
    if cfg.fix_fc_covar:
        with tf.name_scope("fix_ekf_covar"):
            fc_outputs_shape = fc_outputs.get_shape().as_list()
            fixed_covar = np.stack([cfg.fc_covar_fix_val] * fc_outputs_shape[1])
            fixed_covar = np.stack([fixed_covar] * fc_outputs_shape[0])
            fc_outputs = tf.concat([fc_outputs[:, :, 0:6], tf.constant(fixed_covar, tf.float32)], axis=2)

    with tf.name_scope("stack_for_ekf"):
        stack1 = []
        for i in range(fc_outputs.shape[0]):
            stack2 = []
            for j in range(fc_outputs.shape[1]):
                stack2.append(tf.diag(tf.square(fc_outputs[i, j, 6:]) + np.array([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8])))
            stack1.append(tf.stack(stack2, axis=0))

        nn_covar = tf.stack(stack1, axis=0)

    if use_ekf:
        print("Building EKF...")
        # at this point the outputs from the fully connected layer are  [x, y, z, yaw, pitch, roll, 6 x covars]

        with tf.variable_scope("imu_noise_params", reuse=True):
            gyro_bias_diag = tf.get_variable('gyro_bias_sqrt')
            acc_bias_diag = tf.get_variable('acc_bias_sqrt')
            gyro_covar_diag = tf.get_variable('gyro_sqrt')
            acc_covar_diag = tf.get_variable('acc_sqrt')

        with tf.name_scope("ekf_ops"):
            gyro_bias_covar = tf.diag(tf.square(gyro_bias_diag) + 1e-5)
            acc_bias_covar = tf.diag(tf.square(acc_bias_diag) + 1e-5)
            gyro_covar = tf.diag(tf.square(gyro_covar_diag) + 1e-5)
            acc_covar = tf.diag(tf.square(acc_covar_diag) + 1e-5)

            ekf_out_states, ekf_out_covar = ekf.full_ekf_layer(imu_data, fc_outputs[..., 0:6], nn_covar,
                                                               feed_ekf_init_state, feed_ekf_init_covar,
                                                               gyro_bias_covar, acc_bias_covar, gyro_covar, acc_covar)

            rel_disp = tf.concat([ekf_out_states[1:, :, 0:3], ekf_out_states[1:, :, 11:14]], axis=-1)
            rel_covar = tf.concat(
                    [tf.concat([ekf_out_covar[1:, :, 0:3, 0:3], ekf_out_covar[1:, :, 0:3, 11:14]], axis=-1),
                     tf.concat([ekf_out_covar[1:, :, 11:14, 0:3], ekf_out_covar[1:, :, 11:14, 11:14]],
                               axis=-1)], axis=-2)
    else:
        with tf.name_scope("ekf_ops"):
            rel_disp = fc_outputs[..., 0:6]
            rel_covar = nn_covar

            ekf_out_states = tf.zeros([fc_outputs.shape[0], fc_outputs.shape[1], 17], dtype=tf.float32)
            ekf_out_covar = tf.eye(17, batch_shape=[fc_outputs.shape[0], fc_outputs.shape[1]], dtype=tf.float32)

    print("Building SE3...")
    # at this point the outputs are the relative states with covariance, need to only select the part the
    # loss cares about

    se3_outputs = se3_layer(rel_disp, initial_poses)

    return rel_disp, rel_covar, se3_outputs, lstm_states, \
           ekf_out_states[-1, ...], ekf_out_covar[-1, ...], \
           feed_init_states, feed_ekf_init_state, feed_ekf_init_covar
