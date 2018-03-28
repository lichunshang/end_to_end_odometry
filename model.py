import tensorflow as tf
import se3
import tools
import config as cfg


# CNN Block
def cnn_model(inputs):
    with tf.variable_scope("cnn_model"):
        conv_1 = tf.contrib.layers.conv2d(inputs, num_outputs=64, kernel_size=(7, 7,),
                                          stride=(2, 2), padding="same", scope="conv_1", data_format="NCHW")
        conv_2 = tf.contrib.layers.conv2d(conv_1, num_outputs=128, kernel_size=(5, 5,),
                                          stride=(2, 2), padding="same", scope="conv_2", data_format="NCHW")

        conv_3 = tf.contrib.layers.conv2d(conv_2, num_outputs=256, kernel_size=(5, 5,),
                                          stride=(2, 2), padding="same", scope="conv_3", data_format="NCHW")
        conv_3_1 = tf.contrib.layers.conv2d(conv_3, num_outputs=256, kernel_size=(3, 3,),
                                            stride=(1, 1), padding="same", scope="conv_3_1", data_format="NCHW")

        conv_4 = tf.contrib.layers.conv2d(conv_3_1, num_outputs=512, kernel_size=(3, 3,),
                                          stride=(2, 2), padding="same", scope="conv_4", data_format="NCHW")
        conv_4_1 = tf.contrib.layers.conv2d(conv_4, num_outputs=512, kernel_size=(3, 3,),
                                            stride=(1, 1), padding="same", scope="conv_4_1", data_format="NCHW")

        conv_5 = tf.contrib.layers.conv2d(conv_4_1, num_outputs=512, kernel_size=(3, 3,),
                                          stride=(2, 2), padding="same", scope="conv_5", data_format="NCHW")
        conv_5_1 = tf.contrib.layers.conv2d(conv_5, num_outputs=512, kernel_size=(3, 3,),
                                            stride=(1, 1), padding="same", scope="conv_5_1", data_format="NCHW")

        conv_6 = tf.contrib.layers.conv2d(conv_5_1, num_outputs=1024, kernel_size=(3, 3,),
                                          stride=(2, 2), padding="same", scope="conv_6", data_format="NCHW")
        return conv_6

def cnn_model_lidar(inputs):
    with tf.variable_scope("cnn_model"):
        # The first kernel is a 1d convolution
        conv_1 = tf.contrib.layers.conv2d(inputs, num_outputs=64, kernel_size=(7, 7,),
                                          stride=(2, 2), padding="same", scope="conv_1", data_format="NCHW")
        conv_2 = tf.contrib.layers.conv2d(conv_1, num_outputs=128, kernel_size=(5, 5,),
                                          stride=(2, 2), padding="same", scope="conv_2", data_format="NCHW")

        conv_3 = tf.contrib.layers.conv2d(conv_2, num_outputs=256, kernel_size=(5, 5,),
                                          stride=(2, 2), padding="same", scope="conv_3", data_format="NCHW")
        conv_3_1 = tf.contrib.layers.conv2d(conv_3, num_outputs=256, kernel_size=(3, 3,),
                                            stride=(1, 1), padding="same", scope="conv_3_1", data_format="NCHW")

        conv_4 = tf.contrib.layers.conv2d(conv_3_1, num_outputs=512, kernel_size=(3, 3,),
                                          stride=(2, 2), padding="same", scope="conv_4", data_format="NCHW")
        conv_4_1 = tf.contrib.layers.conv2d(conv_4, num_outputs=512, kernel_size=(3, 3,),
                                            stride=(1, 1), padding="same", scope="conv_4_1", data_format="NCHW")

        conv_5 = tf.contrib.layers.conv2d(conv_4_1, num_outputs=512, kernel_size=(3, 3,),
                                          stride=(2, 2), padding="same", scope="conv_5", data_format="NCHW")
        conv_5_1 = tf.contrib.layers.conv2d(conv_5, num_outputs=512, kernel_size=(3, 3,),
                                            stride=(1, 1), padding="same", scope="conv_5_1", data_format="NCHW")

        conv_6 = tf.contrib.layers.conv2d(conv_5_1, num_outputs=1024, kernel_size=(3, 3,),
                                          stride=(2, 2), padding="same", scope="conv_6", data_format="NCHW", activation_fn=None)
        return conv_6

def fc_model(inputs):
    with tf.variable_scope("fc_model"):
        fc_128 = tf.contrib.layers.fully_connected(inputs, 128, scope="fc_128", activation_fn=tf.nn.relu)
        fc_12 = tf.contrib.layers.fully_connected(fc_128, 12, scope="fc_12", activation_fn=None)
        return fc_12


def cnn_over_timesteps(inputs):
    with tf.variable_scope("cnn_over_timesteps"):
        unstacked_inputs = tf.unstack(inputs, axis=0)

        outputs = []

        for i in range(len(unstacked_inputs) - 1):
            # stack images along channels
            image_stacked = tf.concat((unstacked_inputs[i], unstacked_inputs[i + 1]), axis=1)
            outputs.append(cnn_model(image_stacked))

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


def cnn_layer(inputs):
    with tf.variable_scope("cnn_layer", reuse=tf.AUTO_REUSE):
        outputs = cnn_over_timesteps(inputs)

    outputs = tf.reshape(outputs,
                         [outputs.shape[0], outputs.shape[1], outputs.shape[2] * outputs.shape[3] * outputs.shape[4]])

    return outputs


def rnn_layer(inputs, initial_state):
    with tf.variable_scope("rnn_layer", reuse=tf.AUTO_REUSE):
        initial_state = tuple(tf.unstack(initial_state))

        lstm = tf.contrib.cudnn_rnn.CudnnLSTM(cfg.lstm_layers, cfg.lstm_size)
        outputs, final_state = lstm(inputs, initial_state=initial_state)
        return outputs, final_state


def fc_layer(inputs):
    with tf.variable_scope("fc_layer", reuse=tf.AUTO_REUSE):
        fc_outputs = tools.static_map_fn(fc_model, inputs, axis=0)

    return fc_outputs


def se3_layer(inputs, initial_poses):
    with tf.variable_scope("se3_layer", reuse=tf.AUTO_REUSE):
        unstacked_inputs = tf.unstack(inputs, axis=1)
        unstacked_initial_poses = tf.unstack(initial_poses, axis=1)

        outputs = []

        for fc_timesteps, initial_pose in zip(unstacked_inputs, unstacked_initial_poses):
            outputs.append(se3_comp_over_timesteps(fc_timesteps, initial_pose))

        return tf.stack(outputs, axis=1)


def build_training_model(inputs, lstm_initial_state, initial_poses):
    print("Building training model")

    print("Building CNN...")
    with tf.device("/gpu:0"):
        cnn_outputs = cnn_layer(inputs)

    print("Building RNN...")
    with tf.device("/gpu:0"):
        lstm_outputs, lstm_states = rnn_layer(cnn_outputs, lstm_initial_state)

    print("Building FC...")
    with tf.device("/gpu:0"):
        fc_outputs = fc_layer(lstm_outputs)

    print("Building SE3...")
    with tf.device("/gpu:0"):
        # at this point the outputs from the fully connected layer are  [x, y, z, yaw, pitch, roll, 6 x covars]
        se3_outputs = se3_layer(fc_outputs, initial_poses)

    return fc_outputs, se3_outputs, lstm_states
