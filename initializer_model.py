import tensorflow as tf
import model as model

def model_inputs(cfg):
    # All time major
    inputs = tf.placeholder(tf.float32, name="inputs",
                            shape=[cfg.init_steps + 1, cfg.batch_size, cfg.input_channels, cfg.input_height,
                                   cfg.input_width])

    # is training
    is_training = tf.placeholder(tf.bool, name="is_training", shape=[])

    return inputs, is_training

def cnn_over_timesteps(inputs, is_training):
    with tf.variable_scope("cnn_over_timesteps"):
        unstacked_inputs = tf.unstack(inputs, axis=0)

        outputs = []

        for i in range(len(unstacked_inputs) - 1):
            # stack images along channels
            image_stacked = tf.concat((unstacked_inputs[i], unstacked_inputs[i + 1]), axis=1)
            outputs.append(model.cnn_model_lidar(image_stacked, is_training, False))

        return tf.concat(tf.stack(outputs, axis=0), axis=0)

def cnn_layer(inputs, is_training):
    with tf.variable_scope("cnn_layer", reuse=tf.AUTO_REUSE):
        outputs = cnn_over_timesteps(inputs, is_training)

    outputs = tf.reshape(outputs,
                         [outputs.shape[0], outputs.shape[1],
                          outputs.shape[2] * outputs.shape[3] * outputs.shape[4]])

    return outputs


def build_init_model(cfg, get_activations=False):
    print("Building sequence to sequence training model")

    inputs, is_training = model_inputs(cfg)

    print("Building CNN...")
    cnn_outputs = cnn_layer(inputs, is_training)

    print("Building FC...")
    fullyconnected = tf.contrib.layers.fully_connected(inputs, cfg.lstm_size * cfg.lstm_layers * 2, scope="fc", activation_fn=tf.nn.tanh)
    lstm_initial_state = tf.reshape(fullyconnected, 2, cfg.lstm_layers, cfg.lstm_size)

    return inputs, lstm_initial_state, is_training