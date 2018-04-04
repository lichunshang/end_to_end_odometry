import data
import tensorflow as tf

with tf.device("/cpu:0"):
    data.StatefulDataGen("/home/cs4li/Dev/KITTI/dataset/", ["00"])
