import tensorflow as tf
from se3 import *

def fc_losses():

    q1 = tf.constant([[[1, 0, 0, 0]]], dtype=tf.float32)
    q2 = tf.constant([[[0.99996, 0.00873, 0, 0]]], dtype=tf.float32)

    sampling = tf.constant([-1, -0.5, 0, 0.5, 1], dtype=tf.float32)

    result = tf.acos(sampling)

    diff = quat_subtract(q1, q2)

    with tf.Session() as sess:
        print("q1", q1.eval())
        print("q2", q2.eval())
        print("diff", diff.eval())
        print("sampling", sampling.eval())
        print("result", result.eval())

fc_losses()
