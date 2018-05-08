import tensorflow as tf
import se3
import tools
import config
import native_lstm

# CNN Block
# is_training to control whether to apply dropout
def cnn_model(inputs, is_training):
    with tf.variable_scope("cnn_model"):
        conv_1 = tf.contrib.layers.conv2d(inputs, num_outputs=64, kernel_size=(7, 7,),
                                          stride=(2, 2), padding="same", scope="conv_1", data_format="NCHW",
                                          weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                          biases_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0004),
                                          activation_fn=tf.nn.leaky_relu)
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
    with tf.variable_scope("fc_model"):
        fc_128 = tf.contrib.layers.fully_connected(inputs, 128, scope="fc_128", activation_fn=tf.nn.relu)
        fc_12 = tf.contrib.layers.fully_connected(fc_128, 12, scope="fc_12", activation_fn=None)
        return fc_12


def pair_train_fc_layer(inputs):
    with tf.variable_scope("pair_train_fc_model", reuse=tf.AUTO_REUSE):
        fc_128 = tf.contrib.layers.fully_connected(inputs, 128, scope="fc_128", activation_fn=tf.nn.relu,
                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0005))
        fc_6 = tf.contrib.layers.fully_connected(fc_128, 6, scope="fc_6", activation_fn=None)
        return fc_6


def pair_train_fc_layer_1024(inputs):
    with tf.variable_scope("pair_train_fc_model", reuse=tf.AUTO_REUSE):
        fc_128 = tf.contrib.layers.fully_connected(inputs, 1024, scope="fc_128", activation_fn=tf.nn.relu)
        fc_6 = tf.contrib.layers.fully_connected(fc_128, 6, scope="fc_6", activation_fn=None)
        return fc_6


def cnn_over_timesteps(inputs, is_training, get_activations):
    with tf.variable_scope("cnn_over_timesteps"):
        unstacked_inputs = tf.unstack(inputs, axis=0)

        outputs = []

        for i in range(len(unstacked_inputs) - 1):
            # stack images along channels
            image_stacked = tf.concat((unstacked_inputs[i], unstacked_inputs[i + 1]), axis=1)
            outputs.append(cnn_model_lidar(image_stacked, is_training, get_activations))

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


def cnn_layer(inputs, is_training, get_activations):
    with tf.variable_scope("cnn_layer", reuse=tf.AUTO_REUSE):
        outputs = cnn_over_timesteps(inputs, is_training, get_activations)

    outputs = tf.reshape(outputs,
                         [outputs.shape[0], outputs.shape[1], outputs.shape[2] * outputs.shape[3] * outputs.shape[4]])

    return outputs


def rnn_layer(cfg, inputs, initial_state):
    with tf.variable_scope("rnn_layer", reuse=tf.AUTO_REUSE):
        initial_state = tuple(tf.unstack(initial_state))

        #need to break up LSTMs to get states in the middle
        mid_offset = cfg.sequence_stride
        if mid_offset < inputs.shape[0]:
            lstm1 = native_lstm.CudnnLSTM(cfg.lstm_layers, cfg.lstm_size, name='rnn_layer', variable_namespace="rnn_layer")
            lstm1.build(inputs[0:mid_offset, :, :].shape)
            mid_output, output_state = lstm1(inputs[0:mid_offset, :, :], initial_state=initial_state, training=True)

            lstm2 = native_lstm.CudnnLSTM(cfg.lstm_layers, cfg.lstm_size, name='rnn_layer', variable_namespace="rnn_layer")
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


def model_inputs(cfg):
    # All time major
    inputs = tf.placeholder(tf.float32, name="inputs",
                            shape=[cfg.timesteps + 1, cfg.batch_size, cfg.input_channels, cfg.input_height,
                                   cfg.input_width])

    # accommodate the pairwise training
    if hasattr(cfg, "lstm_layers"):
        # init LSTM states, 2 (cell + hidden states), 2 layers, batch size, and 1024 state size
        lstm_initial_state = tf.placeholder(tf.float32, name="lstm_init_state",
                                            shape=[2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size])
    else:
        lstm_initial_state = None

    # init poses, initial position for each example in the batch
    initial_poses = tf.placeholder(tf.float32, name="initial_poses", shape=[cfg.batch_size, 7])

    # is training
    is_training = tf.placeholder(tf.bool, name="is_training", shape=[])

    return inputs, lstm_initial_state, initial_poses, is_training


def model_labels(cfg):
    # 7 for translation + quat
    se3_labels = tf.placeholder(tf.float32, name="se3_labels", shape=[cfg.timesteps, cfg.batch_size, 7])

    # 6 for translation + rpy, labels not needed for covars
    fc_labels = tf.placeholder(tf.float32, name="se3_labels", shape=[cfg.timesteps, cfg.batch_size, 6])

    return se3_labels, fc_labels

def build_seq_model(cfg, get_activations=False, use_initializer=False):
    print("Building sequence to sequence training model")

    if use_initializer:
        inputs, _, initial_poses, is_training = model_inputs(cfg)
    else:
        inputs, lstm_initial_state, initial_poses, is_training = model_inputs(cfg)

    print("Building CNN...")
    cnn_outputs = cnn_layer(inputs, is_training, get_activations)

    if use_initializer:
        print("Building Initializer Network")
        init_list = tf.unstack(cnn_outputs[:cfg.init_length, ...], axis=0)
        init_feed = tf.concat(init_list, axis=1)
        initializer = tf.contrib.layers.fully_connected(init_feed, cfg.lstm_size * cfg.lstm_layers * 2, scope="fc",
                                                           activation_fn=tf.nn.tanh)
        lstm_initial_state = tf.reshape(initializer, [2, cfg.lstm_layers, cfg.batch_size, cfg.lstm_size])

    print("Building RNN...")
    lstm_outputs, lstm_states = rnn_layer(cfg, cnn_outputs, lstm_initial_state)

    print("Building FC...")
    fc_outputs = fc_layer(lstm_outputs, pair_train_fc_layer)

    print("Building SE3...")
    # at this point the outputs from the fully connected layer are  [x, y, z, yaw, pitch, roll, 6 x covars]
    se3_outputs = se3_layer(fc_outputs, initial_poses)

    return inputs, lstm_initial_state, initial_poses, is_training, fc_outputs, se3_outputs, lstm_states

def build_model_w_init(cfg, get_activations=False):
    print("Building S2S train mdl w/ init network")

    inputs, lstm_initial_state, initial_poses, is_training = model_inputs(cfg)

    print("Building CNN")
    cnn_outputs = cnn_layer(inputs, is_training, get_activations)

def build_pair_model(cfg):
    print("Building sequence to sequence training model")

    inputs, _, _, is_training = model_inputs(cfg)

    print("Building CNN...")
    
    cnn_outputs = cnn_layer(inputs, is_training)

    print("Building FC...")
    
    fc_outputs = fc_layer(cnn_outputs, pair_train_fc_layer)

    return inputs, is_training, fc_outputs
