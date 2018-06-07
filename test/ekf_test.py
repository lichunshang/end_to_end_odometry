from __future__ import absolute_import, division, print_function
import tensorflow as tf

tf.enable_eager_execution()

import ekf
import unittest

def no_movement_sample_data():
    imu_meas = tf.constant([[[0, 0, 0, 0, 0, 9.80665]]], dtype=tf.float32)
    nn_meas = tf.constant([[[0, 0, 0, 0, 0, 0]]], dtype=tf.float32)
    nn_covar = tf.expand_dims(tf.expand_dims(tf.eye(6, dtype=tf.float32), axis=0), axis=0)
    prev_state = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)
    prev_covar = tf.expand_dims(tf.eye(17, dtype=tf.float32), axis=0)
    gyro_bias_covar = tf.eye(3, dtype=tf.float32)
    acc_bias_covar = tf.eye(3, dtype=tf.float32)
    gyro_covar = tf.eye(3, dtype=tf.float32)
    acc_covar = tf.eye(3, dtype=tf.float32)

    return imu_meas, nn_meas, nn_covar, prev_state, prev_covar, gyro_bias_covar, acc_bias_covar, gyro_covar, acc_covar

def test_ekf_update():
    imu_meas, nn_meas, nn_covar, prev_state, prev_covar, gyro_bias_covar, acc_bias_covar, gyro_covar, acc_covar = no_movement_sample_data()
    ekf.full_ekf_layer_eager(imu_meas, nn_meas, nn_covar, prev_state, prev_covar, gyro_bias_covar, acc_bias_covar, gyro_covar, acc_covar)
    return True

class EkfTest(unittest.TestCase):
    def test(self):
        self.assertTrue(test_ekf_update())
