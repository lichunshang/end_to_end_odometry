import tensorflow as tf
import math as m

tfe = tf.contrib.eager


# function to map tensors with inner dimension 3 to 3x3 skew symmetric matrices
def skew(x):
    z_el = tf.zeros([x.shape[0], 1, 1])
    x0 = tf.expand_dims(tf.expand_dims(x[..., 0], axis=-1), axis=-1)
    x1 = tf.expand_dims(tf.expand_dims(x[..., 1], axis=-1), axis=-1)
    x2 = tf.expand_dims(tf.expand_dims(x[..., 2], axis=-1), axis=-1)
    return tf.concat((
        tf.concat((z_el, -x2, x1), axis=-1),
        tf.concat((x2, z_el, -x0), axis=-1),
        tf.concat((-x1, x0, z_el), axis=-1)), axis=-2)


# This is a simple ekf layer to fuse angular rate measurements with the network output

# The states of the EKF are:
#   Rotation displacement between current set of frames R(t)
#   Bias of the gyroscope between current set of frames b(t)

# In the process model, the measurements of the IMU are used as input. The IMU noise is thus applied
# to the uncertainty of the process model. The IMU bias is modelled as a slowly changing bias.

# Process model is R(t) = Rimu(t) - delta_T * b(t-1)
#                  b(t) = b(t-1)

# Measurement model is h(x(t)) = R(t)
# Measurement itself is R_NN(t)

# the measurements should have shape timesteps x batches x 3
# prev biases should have shape batches x 3
# covar matrices should be 3x3, except for prev_covar which is batch_size x 6x6 and nn_covar which is time x batch x 6 x 6
def rotation_only_ekf(imu_meas, nn_meas, nn_covar, prev_bias, prev_covar, bias_covar, imu_covar,
                      timestep=0.1):
    with tf.variable_scope("ekf_layer"):
        x_output = []
        covar_output = []

        bias = prev_bias
        sys_covar = prev_covar

        Qkimu = timestep * imu_covar
        Qkbias = timestep * bias_covar

        Fk = tf.eye(6)
        Fk[0:3, 0:3] = 0
        Fk[0:3, 3:6] = -timestep * tf.eye(3)

        Hk = tf.constant([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], name="Hk", dtype=tf.float32)
        Hk = tf.tile(tf.expand_dims(Hk, axis=0), [imu_meas.shape[1]])

        # matmul doesn't support broadcasting so need to repeat Fk and Hk a bunch of times
        Fk = tf.tile(tf.expand_dims(Fk, axis=0), [imu_meas.shape[1]])

        for i in range(imu_meas.shape[0]):
            xpred = tf.concat(imu_meas[i, ...] - timestep * bias, bias, axis=1)
            Pminus = tf.matmul(Fk, tf.matmul(sys_covar, Fk, transpose_b=True))
            Pminus[:, 0:3, 0:3] = Pminus[:, 0:3, 0:3] + Qkimu
            Pminus[:, 3:6, 3:6] = Pminus[:, 3:6, 3:6] + Qkbias

            yt = nn_meas[i, ...] - xpred[:, 0:3]
            Sk = tf.matmul(Hk, tf.matmul(Pminus, Hk, transpose_b=True)) + nn_covar[i, ...]
            Kk = tf.matmul(Pminus, tf.matmul(Hk, tf.matrix_inverse(Sk), transpose_a=True))

            sys_covar = tf.matmul(tf.eye(6, dtype=tf.float32) - tf.matmul(Kk, Hk), Pminus)
            x_final = xpred + tf.matmul(Kk, yt)

            bias = x_final[:, 3:6]

            x_output.append(x_final)
            covar_output.append(sys_covar)

    return tf.stack(x_output), tf.stack(covar_output)


# Function to take euler angles (yaw pitch roll convention) R = R_roll R_pitch R_yaw or XYZ and convert to
# rotation matrices

def euler2rot(eulers):
    sA = tf.sin(eulers[..., 0])
    cA = tf.cos(eulers[..., 0])
    sB = tf.sin(eulers[..., 1])
    cB = tf.cos(eulers[..., 1])
    sC = tf.sin(eulers[..., 2])
    cC = tf.cos(eulers[..., 2])

    # list of output values in row-major form (apparently that is what tensorflow uses when reshaping)
    vals = []
    vals.append(tf.multiply(cB, cA))
    vals.append(tf.multiply(cB, sA))
    vals.append(sB)

    vals.append(-tf.multiply(cC, sA) - tf.multiply(tf.multiply(sC, sB), cA))
    vals.append(tf.multiply(cC, cA) - tf.multiply(tf.multiply(sC, sB), sA))
    vals.append(tf.multiply(sC, cB))

    vals.append(tf.multiply(sC, sA) - tf.multiply(tf.multiply(cC, sB), cA))
    vals.append(-tf.multiply(sC, cA) - tf.multiply(tf.multiply(cC, sB), sA))
    vals.append(tf.multiply(cC, cB))

    tens = tf.stack(vals, axis=-1)
    return tf.reshape(tens, [eulers.shape[0], 3, 3])


# Function to take euler angles (yaw pitch roll convention) R = R_roll R_pitch R_yaw or XYZ and convert to
# rotation matrices yaw is not provided and assumed zero

def euler2rot2param(eulers):
    sA = tf.constant(0, dtype=tf.float32)
    cA = tf.constant(1, dtype=tf.float32)
    sB = tf.sin(eulers[..., 0])
    cB = tf.cos(eulers[..., 0])
    sC = tf.sin(eulers[..., 1])
    cC = tf.cos(eulers[..., 1])

    # list of output values in row-major form (apparently that is what tensorflow uses when reshaping)
    vals = []
    vals.append(tf.multiply(cB, cA))
    vals.append(tf.multiply(cB, sA))
    vals.append(sB)

    vals.append(-tf.multiply(cC, sA) - tf.multiply(tf.multiply(sC, sB), cA))
    vals.append(tf.multiply(cC, cA) - tf.multiply(tf.multiply(sC, sB), sA))
    vals.append(tf.multiply(sC, cB))

    vals.append(tf.multiply(sC, sA) - tf.multiply(tf.multiply(cC, sB), cA))
    vals.append(-tf.multiply(sC, cA) - tf.multiply(tf.multiply(cC, sB), sA))
    vals.append(tf.multiply(cC, cB))

    tens = tf.stack(vals, axis=-1)
    return tf.reshape(tens, [eulers.shape[0], 3, 3])


# function to calculate jacobian of transformation wrt Euler angles

# eulers should have last dimension as 3, with yaw pitch roll ordering
# pt should have last dimension as 3, with x y z ordering
def getJacobian(eulers, ptd):
    sA = tf.sin(eulers[..., 0])
    cA = tf.cos(eulers[..., 0])
    sB = tf.sin(eulers[..., 1])
    cB = tf.cos(eulers[..., 1])
    sG = tf.sin(eulers[..., 2])
    cG = tf.cos(eulers[..., 2])

    # Jacobian is a 3x3. Assembling in row-major form
    pt = tf.squeeze(ptd)
    vals = []
    vals.append(pt[..., 1] * cA * cB - pt[..., 0] * cB * sA)
    vals.append(pt[..., 2] * cB - pt[..., 0] * cA * sB - pt[..., 1] * sA * sB)
    vals.append(tf.zeros([eulers.shape[0]], dtype=tf.float32))

    vals.append(- pt[..., 0] * (cA * cG - sA * sB * sG) - pt[..., 1] * (cG * sA + cA * sB * sG))
    vals.append(- pt[..., 2] * sB * sG - pt[..., 0] * cA * cB * sG - pt[..., 1] * cB * sA * sG)
    vals.append(pt[..., 0] * (sA * sG - cA * cG * sB) - pt[..., 1] * (cA * sG + cG * sA * sB) + pt[..., 2] * cB * cG)

    vals.append(pt[..., 0] * (cA * sG + cG * sA * sB) + pt[..., 1] * (sA * sG - cA * cG * sB))
    vals.append(- pt[..., 2] * cG * sB - pt[..., 0] * cA * cB * cG - pt[..., 1] * cB * cG * sA)
    vals.append(pt[..., 0] * (cG * sA + cA * sB * sG) - pt[..., 1] * (cA * cG - sA * sB * sG) - pt[..., 2] * cB * sG)

    tens = tf.stack(vals, axis=-1)
    return tf.reshape(tens, [eulers.shape[0], 3, 3])


def getLittleJacobian(eulers):
    sB = tf.sin(eulers[..., 0])
    cB = tf.cos(eulers[..., 0])
    sG = tf.sin(eulers[..., 1])
    cG = tf.cos(eulers[..., 1])

    # Jacobian is a 3x2. Assembling in row-major form
    vals = []
    vals.append(cB)
    vals.append(tf.zeros([eulers.shape[0]]))

    vals.append(-sG * sB)
    vals.append(cG * cB)

    vals.append(-cG * sB)
    vals.append(-sG * cB)

    tens = tf.stack(vals, axis=-1)
    return tf.reshape(tens, [eulers.shape[0], 3, 2])


# This is a simple ekf layer to fuse angular rate and linear acceleration measurements with the network output

# The states of the EKF are:
#   Rotation displacement between current set of frames R(t)
#   Bias of the gyroscope between current set of frames bG(t)
#   Bias of the accelerometer between current sets of frames bA(t)
#   Position of the last frame wrt the first frame p(t)
#   Linear velocity in the body frame wrt its position at the start of the set of frames
#   Orientation at the start wrt gravity

# In the process model, the measurements of the IMU are used as input. The IMU noise is thus applied
# to the uncertainty of the process model. The IMU bias is modelled as a slowly changing bias.

# Process model is R(t) = Rimu(t) - dT * bg(t-1)
#                  bg(t) = bg(t-1)
#                  p(t) = (Rimu(t) - dT * bg(t-1)) * v(t-1) * dT + toRot(Rimu(t) - dT * bg(t-1) + Rg(t-1)) * g dT^2/2 + dT^2 / 2 (a_imu(t) - bac(t-1))
#                  v(t) = (Rimu(t) - dT * bg(t-1)) * v(t-1) + toRot(Rimu(t) - dT * bg(t-1) + Rg(t-1)) * g dT + dT * (a_imu(t) - bac(t-1))
#                  Rg(t) = Rimu(t) - dT * bg(t-1) + Rg(t-1)
#                  bac(t) = bac(t-1)

# Measurement model is h(x(t)) = [p(t), R(t)]^T

# the measurements should have shape timesteps x batches x 6. order is yaw, pitch, roll, x, y, z,
# prev biases should have shape batches x 3
# covar matrices should be 3x3, except for prev_covar which is batch_size x 17x17 and nn_covar which is time x batch x 6 x 6

def full_ekf_layer(imu_meas_in, nn_meas, nn_covar, prev_state, prev_covar, gyro_bias_covar, acc_bias_covar, gyro_covar,
                   acc_covar, dt=0.1):
    with tf.variable_scope("ekf_layer", reuse=tf.AUTO_REUSE):
        prev_states = []
        covar_output = []

        prev_states.append(prev_state)
        covar_output.append(prev_covar)

        imu_meas = tf.concat([tf.reverse(imu_meas_in[..., 0:3], axis=[-1]), imu_meas_in[..., 3:6]], axis=-1,
                             name='imu_meas')

        # state update. Reused variables are calculated
        lift_g_covar = tf.tile(tf.expand_dims(gyro_covar, axis=0), [imu_meas.shape[1], 1, 1])
        lift_a_covar = tf.tile(tf.expand_dims(acc_covar, axis=0), [imu_meas.shape[1], 1, 1])
        lift_g_bias_covar = tf.tile(tf.expand_dims(gyro_bias_covar, axis=0), [imu_meas.shape[1], 1, 1])
        lift_a_bias_covar = tf.tile(tf.expand_dims(acc_bias_covar, axis=0), [imu_meas.shape[1], 1, 1])

        g = -9.80665
        gfull = tf.tile(tf.expand_dims(tf.constant([[0], [0], [g]], dtype=tf.float32, name="g"), axis=0),
                        [imu_meas.shape[1], 1, 1])

        diRo = tf.constant([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -dt, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -dt]], dtype=tf.float32, name="diRo")

        dbacc = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=tf.float32, name="dbacc")

        diRim1 = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -dt, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -dt, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -dt]], dtype=tf.float32, name="diRim1")

        dbgy = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=tf.float32, name="dbgy")

        # Prepare the constant part of Fk and tile across all batches
        fkstat = tf.tile(tf.expand_dims(tf.concat([diRo, dbacc, diRim1, dbgy], axis=0), axis=0),
                         [imu_meas.shape[1], 1, 1])

        # hk is mercifully constant
        hk = tf.constant([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=tf.float32, name="hk")

        Hk = tf.tile(tf.expand_dims(hk, axis=0), [imu_meas.shape[1], 1, 1])

        for i in range(imu_meas.shape[0]):
            next_state, next_covariance = run_update(imu_meas[i, ...], dt, prev_states[i], covar_output[i], gfull, g,
                                                     fkstat,
                                                     Hk, lift_g_bias_covar, lift_a_bias_covar, lift_g_covar,
                                                     lift_a_covar, nn_meas[i, ...],
                                                     nn_covar[i, ...])

            prev_states.append(next_state)
            covar_output.append(next_covariance)

    return tf.stack(prev_states), tf.stack(covar_output)


# Abstract out one iteration of the ekf

def run_update(imu_meas_in, dt, prev_state_in, prev_covar_in, gfull, g, fkstat, Hk, lift_g_bias_covar,
               lift_a_bias_covar, lift_g_covar, lift_a_covar, nn_meas_in, nn_covar_in):
    # multiple by one to get names for debugger
    prev_state = tf.multiply(1.0, prev_state_in, name='ekf_prev_state')
    prev_covar = tf.multiply(1.0, prev_covar_in, name='ekf_prev_covar')
    imu_meas = tf.multiply(1.0, imu_meas_in, name='ekf_imu_meas')
    nn_meas = tf.multiply(1.0, nn_meas_in, name='ekf_nn_meas')
    nn_covar = tf.multiply(1.0, nn_covar_in, name='ekf_nn_covar')

    pred_rot_euler = dt * imu_meas[..., 0:3] - dt * prev_state[..., 14:17]
    pred_rot = euler2rot(pred_rot_euler)
    pred_global = dt * imu_meas[..., 1:3] - dt * prev_state[..., 15:17] + prev_state[..., 6:8]

    pred_global_rot = euler2rot2param(pred_global)

    pred_list = []

    # pos = dt * np.dot(pred_rot, x[3:6]) + (0.5 * dt * dt) * (
    #             np.dot(pred_global_rot, gfull) + imu_meas[3:6] + 2 * np.cross(imu_meas[0:3] - x[14:17], x[3:6]) - x[8:11])

    # position prediction
    pred_list.append(
            tf.squeeze(tf.matmul(pred_rot, dt * tf.expand_dims(prev_state[..., 3:6], axis=-1))) + (0.5 * dt * dt) * \
            (tf.squeeze(tf.matmul(pred_global_rot, gfull)) + imu_meas[:, 3:6] - prev_state[..., 8:11]))
    # velocity prediction
    pred_list.append(tf.squeeze(tf.matmul(pred_rot, tf.expand_dims(prev_state[..., 3:6], axis=-1))) + dt * \
                     (tf.squeeze(tf.matmul(pred_global_rot, gfull)) + imu_meas[:, 3:6] - prev_state[..., 8:11]))

    # global pitch and roll prediction
    pred_list.append(pred_global)

    # accelerometer bias prediction
    pred_list.append(prev_state[..., 8:11])

    # relative orientation prediction
    pred_list.append(pred_rot_euler)

    # gyro bias update
    pred_list.append(prev_state[..., 14:17])

    # pack state
    pred_state = tf.concat(pred_list, axis=1, name="ekf_pred_state")

    # build Jacobian
    dRglobal_dE = tf.concat([tf.zeros([imu_meas.shape[0], 3, 1], dtype=tf.float32), getLittleJacobian(pred_global)],
                            axis=-1)

    Fk = tf.concat([tf.concat([tf.zeros([imu_meas.shape[0], 3, 3]),
                               dt * pred_rot,
                               0.5 * dt * dt * g * getLittleJacobian(pred_global),
                               -0.5 * dt * dt * tf.eye(3, batch_shape=[imu_meas.shape[0]], dtype=tf.float32),
                               tf.zeros([imu_meas.shape[0], 3, 3]),
                               -dt * getJacobian(pred_rot_euler, dt * prev_state[..., 3:6]) -
                               g * 0.5 * dt * dt * dt * dRglobal_dE], axis=-1),
                    tf.concat([tf.zeros([imu_meas.shape[0], 3, 3]),
                               pred_rot,
                               dt * g * getLittleJacobian(pred_global),
                               -dt * tf.eye(3, batch_shape=[imu_meas.shape[0]], dtype=tf.float32),
                               tf.zeros([imu_meas.shape[0], 3, 3]),
                               -dt * getJacobian(pred_rot_euler, prev_state[..., 3:6]) + \
                               -g * dt * dt * dRglobal_dE], axis=-1)
                    ], axis=1)

    Fkfull = tf.concat([Fk, fkstat], axis=1)

    # Combine covariance matrices into one large matrix. order is measurement noise (gyro then acc), then bias noise (
    # gyro then acc)
    noise_covar = tf.concat((tf.concat((lift_g_covar, tf.zeros([imu_meas.shape[0], 3, 9], dtype=tf.float32)), axis=-1),
                             tf.concat((tf.zeros([imu_meas.shape[0], 3, 3], dtype=tf.float32), lift_a_covar,
                                        tf.zeros([imu_meas.shape[0], 3, 6], dtype=tf.float32)), axis=-1),
                             tf.concat((tf.zeros([imu_meas.shape[0], 3, 6], dtype=tf.float32), lift_g_bias_covar,
                                        tf.zeros([imu_meas.shape[0], 3, 3], dtype=tf.float32)), axis=-1),
                             tf.concat((tf.zeros([imu_meas.shape[0], 3, 9], dtype=tf.float32), lift_a_bias_covar),
                                       axis=-1)), axis=-2)

    reuse = tf.concat((tf.zeros([imu_meas.shape[0], 3, 1], dtype=tf.float32), getLittleJacobian(pred_global)), axis=-1)

    dpi = tf.concat((getJacobian(pred_rot_euler, dt * prev_state[..., 3:6]) * -dt
                     + (0.5 * dt * dt) * (-dt * g * reuse),
                     (-0.5 * dt * dt) * tf.eye(3, batch_shape=[imu_meas.shape[0]], dtype=tf.float32),
                     tf.zeros([imu_meas.shape[0], 3, 3], dtype=tf.float32),
                     tf.zeros([imu_meas.shape[0], 3, 3], dtype=tf.float32)), axis=-1)

    dvi = tf.concat((getJacobian(pred_rot_euler, prev_state[..., 3:6]) * -dt
                     + dt * (-dt * g * reuse),
                     (-dt * tf.eye(3, batch_shape=[imu_meas.shape[0]], dtype=tf.float32)),
                     tf.zeros([imu_meas.shape[0], 3, 3], dtype=tf.float32),
                     tf.zeros([imu_meas.shape[0], 3, 3], dtype=tf.float32)), axis=-1)

    drotglobal = tf.tile(tf.expand_dims(
            tf.constant([[0, -dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, -dt, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32, name="dro"), axis=0),
            [imu_meas.shape[0], 1, 1])

    daccbias = tf.tile(tf.expand_dims(tf.constant([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=tf.float32, name="daccbias"), axis=0), [imu_meas.shape[0], 1, 1])

    drotrel = tf.tile(tf.expand_dims(tf.constant([
        [-dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, -dt, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, -dt, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32, name="drotrel"), axis=0), [imu_meas.shape[0], 1, 1])

    dgyrobias = tf.tile(tf.expand_dims(tf.constant([
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], dtype=tf.float32, name="dgyrobias"), axis=0), [imu_meas.shape[0], 1, 1])

    J_noise = tf.concat((dpi, dvi, drotglobal, daccbias, drotrel, dgyrobias), axis=-2)

    # Assemble global covariance matrix

    Qk = tf.matmul(J_noise, tf.matmul(noise_covar, J_noise, transpose_b=True), name="ekf_noise_covar")

    tf.assert_positive(tf.matrix_determinant(Qk), [Qk, J_noise, noise_covar])

    pred_covar = tf.add(tf.matmul(Fkfull, tf.matmul(prev_covar, Fkfull, transpose_b=True)), Qk, name="ekf_pred_covar")

    tf.assert_positive(tf.matrix_determinant(pred_covar), [Fkfull])

    yk = tf.subtract(tf.expand_dims(nn_meas, axis=-1), tf.matmul(Hk, tf.expand_dims(pred_state, axis=-1)),
                     name="ekf_error")

    tf.assert_positive(tf.matrix_determinant(nn_covar), [nn_covar])

    Sk = tf.add(tf.matmul(Hk, tf.matmul(pred_covar, Hk, transpose_b=True)), nn_covar, name="ekf_innovation_covar")

    tf.assert_positive(tf.matrix_determinant(Sk), [Sk])

    Kk = tf.matmul(pred_covar, tf.matmul(Hk, tf.matrix_inverse(Sk), transpose_a=True), name="ekf_gain")

    X = tf.squeeze(tf.expand_dims(pred_state, axis=-1) + tf.matmul(Kk, yk), axis=2, name="ekf_updated_state")

    covar = pred_covar - tf.matmul(Kk, tf.matmul(Hk, pred_covar), name="ekf_updated_covar")

    tf.assert_positive(tf.matrix_determinant(covar), [imu_meas, prev_state, prev_covar, nn_meas, nn_covar])

    return X, covar
