from __future__ import absolute_import, division, print_function
import tensorflow as tf

tf.enable_eager_execution()

import ekf
import unittest

import numpy as np
import numdifftools as nd

#function to map 3 vector to skew symmetric matrix/ces
def skew(x):
    mat = np.array([[0, -x[2], x[1]],
                    [x[2], 0, -x[0]],
                    [-x[1], x[0], 0]], dtype=np.float32)
    return mat

# Function to take euler angles (yaw pitch roll convention) R = R_roll R_pitch R_yaw or XYZ and convert to
# rotation matrices

def euler2rot(eulers):
    sA = np.sin(eulers[0])
    cA = np.cos(eulers[0])
    sB = np.sin(eulers[1])
    cB = np.cos(eulers[1])
    sC = np.sin(eulers[2])
    cC = np.cos(eulers[2])

    # list of output values in row-major form (apparently that is what tensorflow uses when reshaping)
    vals = []
    vals.append(np.multiply(cB, cA))
    vals.append(np.multiply(cB, sA))
    vals.append(sB)

    vals.append(-np.multiply(cC, sA) - np.multiply(np.multiply(sC, sB), cA))
    vals.append(np.multiply(cC, cA) - np.multiply(np.multiply(sC, sB), sA))
    vals.append(np.multiply(sC, cB))

    vals.append(np.multiply(sC, sA) - np.multiply(np.multiply(cC, sB), cA))
    vals.append(-np.multiply(sC, cA) - np.multiply(np.multiply(cC, sB), sA))
    vals.append(np.multiply(cC, cB))

    tens = np.stack(vals, axis=-1)
    return np.reshape(tens, [3, 3])


def euler2rot2param(eulers):
    sA = 0.0
    cA = 1.0
    sB = np.sin(eulers[0])
    cB = np.cos(eulers[0])
    sC = np.sin(eulers[1])
    cC = np.cos(eulers[1])

    # list of output values in row-major form (apparently that is what tensorflow uses when reshaping)
    vals = []
    vals.append(np.multiply(cB, cA))
    vals.append(np.multiply(cB, sA))
    vals.append(sB)

    vals.append(-np.multiply(cC, sA) - np.multiply(np.multiply(sC, sB), cA))
    vals.append(np.multiply(cC, cA) - np.multiply(np.multiply(sC, sB), sA))
    vals.append(np.multiply(sC, cB))

    vals.append(np.multiply(sC, sA) - np.multiply(np.multiply(cC, sB), cA))
    vals.append(-np.multiply(sC, cA) - np.multiply(np.multiply(cC, sB), sA))
    vals.append(np.multiply(cC, cB))

    tens = np.stack(vals, axis=-1)
    return np.reshape(tens, [3, 3])

def getJacobian(eulers, pt):
    sA = np.sin(eulers[0])
    cA = np.cos(eulers[0])
    sB = np.sin(eulers[1])
    cB = np.cos(eulers[1])
    sG = np.sin(eulers[2])
    cG = np.cos(eulers[2])

    # Jacobian is a 3x3. Assembling in row-major form
    vals = []
    vals.append(pt[1]*cA*cB - pt[0]*cB*sA)
    vals.append(pt[2]*cB - pt[0]*cA*sB - pt[1]*sA*sB)
    vals.append(0.0)

    vals.append(- pt[0]*(cA*cG - sA*sB*sG) - pt[1]*(cG*sA + cA*sB*sG))
    vals.append(- pt[2]*sB*sG - pt[0]*cA*cB*sG - pt[1]*cB*sA*sG)
    vals.append(pt[0]*(sA*sG - cA*cG*sB) - pt[1]*(cA*sG + cG*sA*sB) + pt[2]*cB*cG)

    vals.append(pt[0]*(cA*sG + cG*sA*sB) + pt[1]*(sA*sG - cA*cG*sB))
    vals.append(- pt[2]*cG*sB - pt[0]*cA*cB*cG - pt[1]*cB*cG*sA)
    vals.append(pt[0]*(cG*sA + cA*sB*sG) - pt[1]*(cA*cG - sA*sB*sG) - pt[2]*cB*sG)

    tens = np.stack(vals, axis=-1)
    return np.reshape(tens, [3, 3])

def getLittleJacobian(eulers):
    sB = np.sin(eulers[0])
    cB = np.cos(eulers[0])
    sG = np.sin(eulers[1])
    cG = np.cos(eulers[1])

    # Jacobian is a 3x2. Assembling in row-major form
    vals = []
    vals.append(cB)
    vals.append(0.0)

    vals.append(-sG * sB)
    vals.append(cG * cB)

    vals.append(-cG * sB)
    vals.append(-sG * cB)

    tens = np.stack(vals, axis=-1)
    return np.reshape(tens, [3, 2])

imu_meas = np.array([0.1, 0, 0, 0, 0, 9.80665], dtype=np.float32)
gfull = np.array([0, 0, -9.80665], dtype=np.float32)
g = -9.80665
dt = 0.1

# functions to check analytical derivatives against. the state is packed into a vector
def pred_state(x):
    pred_rot_euler = dt * imu_meas[0:3] - dt * x[14:17]
    pred_rot = euler2rot(pred_rot_euler)

    pred_global_euler = dt * imu_meas[1:3] - dt * x[15:17] + x[6:8]
    pred_global_rot = euler2rot2param(pred_global_euler)

    pos = np.dot(pred_rot, dt * x[3:6]) + (0.5 * dt * dt) * (np.dot(pred_global_rot, gfull) + imu_meas[3:6] - x[8:11])
    # velocity prediction
    vel = np.dot(pred_rot, x[3:6]) + dt * (np.dot(pred_global_rot, gfull) + imu_meas[3:6] - x[8:11])

    # global pitch and roll prediction
    glob_rot = pred_global_euler

    # accelerometer bias prediction
    acc_bias = x[8:11]

    # relative orientation prediction
    rot_rel = pred_rot_euler

    # gyro bias update
    gyro_bias = x[14:17]

    # pack state
    return np.concatenate((pos, vel, glob_rot, acc_bias, rot_rel, gyro_bias))

def pred_state_with_noise(x, n):
    pred_rot_euler = dt * imu_meas[0:3] - dt * x[14:17] - dt * n[0:3]
    pred_rot = euler2rot(pred_rot_euler)

    pred_global_euler = dt * imu_meas[1:3] - dt * x[15:17] - dt * n[1:3] + x[6:8]
    pred_global_rot = euler2rot2param(pred_global_euler)

    pos = np.dot(pred_rot, dt * x[3:6]) + (0.5 * dt * dt) * (np.dot(pred_global_rot, gfull) + imu_meas[3:6] - x[8:11] - n[3:6])
    # velocity prediction
    vel = np.dot(pred_rot, x[3:6]) + dt * (np.dot(pred_global_rot, gfull) + imu_meas[3:6] - x[8:11] - n[3:6])

    # global pitch and roll prediction
    glob_rot = pred_global_euler

    # accelerometer bias prediction
    acc_bias = x[8:11] + n[9:12]

    # relative orientation prediction
    rot_rel = pred_rot_euler

    # gyro bias update
    gyro_bias = x[14:17] + n[6:9]

    # pack state
    return np.concatenate((pos, vel, glob_rot, acc_bias, rot_rel, gyro_bias))

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

def ekf_update():
    imu_meas, nn_meas, nn_covar, prev_state, prev_covar, gyro_bias_covar, acc_bias_covar, gyro_covar, acc_covar = no_movement_sample_data()
    ekf.full_ekf_layer(imu_meas, nn_meas, nn_covar, prev_state, prev_covar, gyro_bias_covar, acc_bias_covar, gyro_covar, acc_covar)
    return True

def pred_jacobians():
    x_prev = np.zeros([17], dtype=np.float32)
    x_prev[3] = 2.0

    Jnumerical = nd.Jacobian(pred_state)(x_prev)

    diRo = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -dt, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -dt]], dtype=np.float32)

    dbacc = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

    diRim1 = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -dt, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -dt, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -dt]], dtype=np.float32)

    dbgy = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)

    # Prepare the constant part of Fk
    fkstat = np.concatenate((diRo, dbacc, diRim1, dbgy), axis=0)

    pred_rot_euler = dt * imu_meas[0:3] - dt * x_prev[14:17]
    pred_rot = euler2rot(pred_rot_euler)

    pred_global_euler = dt * imu_meas[1:3] - dt * x_prev[15:17] + x_prev[6:8]
    pred_global_rot = euler2rot2param(pred_global_euler)

    g = -9.80665

    dRglobal_dE = np.concatenate((np.zeros([3, 1], dtype=np.float32), getLittleJacobian(pred_global_euler)), axis=-1)

    dpi = np.concatenate((np.zeros([3, 3], dtype=np.float32),
                               dt * pred_rot,
                               0.5 * dt * dt * g * getLittleJacobian(pred_global_euler),
                               -0.5 * dt * dt * np.eye(3, dtype=np.float32),
                               np.zeros([3, 3], dtype=np.float32),
                               -dt * getJacobian(pred_rot_euler, dt * x_prev[3:6]) -
                               g * 0.5 * dt * dt * dt * dRglobal_dE), axis=-1)

    dvi = np.concatenate([tf.zeros([3, 3]),
                               pred_rot,
                               dt * g * getLittleJacobian(pred_global_euler),
                               -dt * np.eye(3, dtype=np.float32),
                               np.zeros([3, 3], dtype=np.float32),
                               -dt * getJacobian(pred_rot_euler, x_prev[3:6]) + \
                               -g * dt * dt * dRglobal_dE], axis=-1)

    Janalytical = np.concatenate((dpi, dvi, fkstat), axis=0)

    error = Janalytical - Jnumerical

    return np.linalg.norm(error)

def noise_jacobians():
    x_prev = np.zeros([17], dtype=np.float32)
    x_prev[3] = 2.0

    eta = np.zeros([12], dtype=np.float32)

    func = lambda n: pred_state_with_noise(x_prev, n)

    Jnumerical = nd.Jacobian(func)(eta)

    pred_rot_euler = dt * imu_meas[0:3] - dt * x_prev[14:17] - dt * eta[0:3]
    pred_rot = euler2rot(pred_rot_euler)

    pred_global_euler = dt * imu_meas[1:3] - dt * x_prev[15:17] - dt * eta[1:3] + x_prev[6:8]
    pred_global_rot = euler2rot2param(pred_global_euler)

    reuse = np.concatenate((np.zeros([3, 1], dtype=np.float32), getLittleJacobian(pred_global_euler)), axis=-1)

    dpi = np.concatenate((getJacobian(pred_rot_euler, dt * x_prev[3:6]) * -dt
                          + (0.5 * dt * dt) * (-dt * g * reuse)
                          + dt * dt * skew(x_prev[3:6]),
                          (-0.5 * dt * dt * np.eye(3, dtype=np.float32)),
                          np.zeros([3, 3], dtype=np.float32),
                          np.zeros([3, 3], dtype=np.float32)), axis=-1)

    dvi = np.concatenate((getJacobian(pred_rot_euler, x_prev[3:6]) * -dt
                          + dt * (-dt * g * reuse) + 2 * dt * skew(x_prev[3:6]),
                          (-dt * np.eye(3, dtype=np.float32)),
                          np.zeros([3, 3], dtype=np.float32),
                          np.zeros([3, 3], dtype=np.float32)), axis=-1)

    drotglobal = np.array([[0, -dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, -dt, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

    daccbias = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)

    drotrel = np.array([[-dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, -dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, -dt, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)

    dgyrobias = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=np.float32)

    Janalytical = np.concatenate((dpi, dvi, drotglobal, daccbias, drotrel, dgyrobias), axis=0)

    squared = np.dot(Janalytical, np.transpose(Janalytical))

    error = Janalytical - Jnumerical

    return np.linalg.norm(error)

class EkfTest(unittest.TestCase):
    def test(self):
        self.assertTrue(ekf_update())

    def test_state_jacobians(self):
        self.assertLess(pred_jacobians(), 1.0e-7)

    def test_noise_jacobians(self):
        self.assertLess(noise_jacobians(), 1.0e-7)
