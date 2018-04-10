import model
import tensorflow as tf



inputs, is_training, fc_outputs = model.build_pair_training_model()

cnn_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "^cnn_layer.*")

cnn_init_tf_saver = tf.train.Saver(cnn_variables)

weights_mapping = {
    "FlowNetS/conv1/weights": "cnn_layer/cnn_over_timesteps/cnn_model/conv_1/weights",
    "FlowNetS/conv1/biases": "cnn_layer/cnn_over_timesteps/cnn_model/conv_1/biases",

    "FlowNetS/conv2/weights": "cnn_layer/cnn_over_timesteps/cnn_model/conv_2/weights",
    "FlowNetS/conv2/biases": "cnn_layer/cnn_over_timesteps/cnn_model/conv_2/biases",

    "FlowNetS/conv3/weights": "cnn_layer/cnn_over_timesteps/cnn_model/conv_3/weights",
    "FlowNetS/conv3/biases": "cnn_layer/cnn_over_timesteps/cnn_model/conv_3/biases",

    "FlowNetS/conv3_1/weights": "cnn_layer/cnn_over_timesteps/cnn_model/conv_3_1/weights",
    "FlowNetS/conv3_1/biases": "cnn_layer/cnn_over_timesteps/cnn_model/conv_3_1/biases",

    "FlowNetS/conv4/weights": "cnn_layer/cnn_over_timesteps/cnn_model/conv_4/weights",
    "FlowNetS/conv4/biases": "cnn_layer/cnn_over_timesteps/cnn_model/conv_4/biases",

    "FlowNetS/conv4_1/weights": "cnn_layer/cnn_over_timesteps/cnn_model/conv_4_1/weights",
    "FlowNetS/conv4_1/biases": "cnn_layer/cnn_over_timesteps/cnn_model/conv_4_1/biases",

    "FlowNetS/conv5/weights": "cnn_layer/cnn_over_timesteps/cnn_model/conv_5/weights",
    "FlowNetS/conv5/biases": "cnn_layer/cnn_over_timesteps/cnn_model/conv_5/biases",

    "FlowNetS/conv5_1/weights": "cnn_layer/cnn_over_timesteps/cnn_model/conv_5_1/weights",
    "FlowNetS/conv5_1/biases": "cnn_layer/cnn_over_timesteps/cnn_model/conv_5_1/biases",

    "FlowNetS/conv6/weights": "cnn_layer/cnn_over_timesteps/cnn_model/conv_6/weights",
    "FlowNetS/conv6/biases": "cnn_layer/cnn_over_timesteps/cnn_model/conv_6/biases",
}

weight_to_tensor_mapping = {}

for v in cnn_variables:
    for key, val in weights_mapping.items():
        if val in v.name:
            weight_to_tensor_mapping[key] = v
            break


tf_saver = tf.train.Saver(weight_to_tensor_mapping)
tf_saver2 = tf.train.Saver(cnn_variables)
flownet_weight_file = "/home/cs4li/git/tf_flownet2/FlowNet2_src/checkpoints/weights/FlowNetS/flownet-S.ckpt-0"

with tf.Session() as sess:
    tf_saver.restore(sess, flownet_weight_file)
    tf_saver2.save(sess, "/home/cs4li/Dev/end_to_end_visual_odometry/results/flownet_weights/flownet_s_weights")