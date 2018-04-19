import data
import config
import model
import tools
import tensorflow as tf
import numpy as np
import os
import transformations
import matplotlib.pyplot as plt

dir_name = "trajectory_results"
kitti_seq = "06"

if kitti_seq in ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]:
    save_ground_truth = False
else:
    save_ground_truth = True

cfg = config.PairCamEvalConfig

tools.printf("Building eval model....")
inputs, is_training, fc_outputs, = model.build_pair_model(cfg)

tools.printf("Loading training data...")
train_data_gen = data.StatefulDataGen(cfg, "/home/cs4li/Dev/KITTI/dataset/", [kitti_seq], frames=[None])

results_dir_path = os.path.join(config.save_path, dir_name)
if not os.path.exists(results_dir_path):
    os.makedirs(results_dir_path)

# ==== Read Model Checkpoints =====
restore_model_file = "/home/cs4li/Dev/end_to_end_visual_odometry/results/" \
                     "train_pair_20180409-23-44-54_75_epochs_0.125_loss/" \
                     "model_epoch_checkpoint-74"

variable_to_load = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "^(cnn_layer|fc_layer).*")
tf_restore_saver = tf.train.Saver(variable_to_load)

with tf.Session() as sess:
    tools.printf("Restoring model weights from %s..." % restore_model_file)
    tf_restore_saver.restore(sess, restore_model_file)

    total_batches = train_data_gen.total_batches()
    tools.printf("Start evaluation loop...")

    prediction = np.zeros([total_batches + 1, 7])
    deltas = np.zeros([total_batches, 6])
    ground_truth = np.zeros([total_batches + 1, 7])
    fig, ax = plt.subplots(nrows=2, ncols=1)

    while train_data_gen.has_next_batch():
        j_batch = train_data_gen.curr_batch()

        # get inputs
        _, _, batch_data, fc_ground_truth, se3_ground_truth = train_data_gen.next_batch()

        # Run training session
        _fc_outputs = sess.run(
            fc_outputs,
            feed_dict={
                inputs: batch_data,
                is_training: False,
            },
        )

        img1 = np.moveaxis(batch_data[0, 0], 0, 2)
        img1 = np.squeeze(img1)
        img2 = np.moveaxis(batch_data[1, 0], 0, 2)
        img2 = np.squeeze(img2)

        ax[0].imshow(img1[..., [2, 1, 0]])
        ax[1].imshow(img2[..., [2, 1, 0]])
        fig.show()

        deltas[j_batch, :] = _fc_outputs[-1, -1]
        ground_truth[j_batch + 1:] = se3_ground_truth[-1, -1]

        if j_batch % 10 == 0:
            tools.printf("Processed %.2f%%" % (train_data_gen.curr_batch() / train_data_gen.total_batches() * 100))

    prediction[0] = np.array([0., 0., 0., 1., 0., 0., 0.], dtype=np.float32)
    ground_truth[0] = np.array([0., 0., 0., 1., 0., 0., 0.], dtype=np.float32)

    tools.printf("Composing trajectory")
    for i in range(0, deltas.shape[0]):
        # prediction[i] = sess.run(se3.se3_comp(tf.constant(prediction[i], dtype=tf.float32),
        #                                       tf.constant(deltas[i], dtype=tf.float32)))

        p1 = transformations.quaternion_matrix(prediction[i, 3:])
        p1[0, 3] = prediction[i, 0]
        p1[1, 3] = prediction[i, 1]
        p1[2, 3] = prediction[i, 2]

        p2 = transformations.euler_matrix(deltas[i, 3], deltas[i, 4], deltas[i, 5], axes="rzyx")
        p2[0, 3] = deltas[i, 0]
        p2[1, 3] = deltas[i, 1]
        p2[2, 3] = deltas[i, 2]

        ret = np.dot(p1, p2)

        translation = transformations.translation_from_matrix(ret)
        quat = transformations.quaternion_from_matrix(ret)

        prediction[i + 1] = np.concatenate([translation, quat])

        if i % 10 == 0:
            tools.printf("Processed %.2f%%" % ((i + 1) / deltas.shape[0] * 100))

    np.save(os.path.join(results_dir_path, "trajectory_" + kitti_seq), prediction)
    if save_ground_truth:
        np.save(os.path.join(results_dir_path, "ground_truth_" + kitti_seq), ground_truth)
