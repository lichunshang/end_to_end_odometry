import data
import tensorflow as tf

with tf.device("/cpu:0"):
    data.StatefulDataGen("/home/lichunshang/Dev/KITTI/dataset/", ["00"])
