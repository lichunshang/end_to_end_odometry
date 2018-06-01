import tensorflow as tf
import math as m


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
def rotation_only_ekf(imu_meas, nn_meas, nn_covar, prev_bias, prev_covar, bias_covar, imu_covar, timestep=tf.constant(0.1, dtype=tf.float32)):
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

        Hk = tf.Variable([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]])
        Hk = tf.tile(tf.expand_dims(Hk, axis=0), [imu_meas.shape[1]])

        # matmul doesn't support broadcasting so need to repeat Fk and Hk a bunch of times
        Fk = tf.tile(tf.expand_dims(Fk, axis=0), [imu_meas.shape[1]])

        for i in range(imu_meas.shape[0]):
            xpred = tf.concat(imu_meas[i,...] - timestep * bias, bias, axis=1)
            Pminus = tf.matmul(Fk, tf.matmul(sys_covar, Fk, transpose_b=True))
            Pminus[:,0:3,0:3] = Pminus[:,0:3,0:3] + Qkimu
            Pminus[:, 3:6, 3:6] = Pminus[:, 3:6, 3:6] + Qkbias

            yt = nn_meas[i,...] - xpred[:, 0:3]
            Sk = tf.matmul(Hk, tf.matmul(Pminus, Hk, transpose_b=True)) + nn_covar[i,...]
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

# function to calculate jacobian of transformation wrt Euler angles

# eulers should have last dimension as 3, with yaw pitch roll ordering
# pt should have last dimension as 3, with x y z ordering
def getJacobian(eulers, pt):
    sA = tf.sin(eulers[..., 0])
    cA = tf.cos(eulers[..., 0])
    sB = tf.sin(eulers[..., 1])
    cB = tf.cos(eulers[..., 1])
    sG = tf.sin(eulers[..., 2])
    cG = tf.cos(eulers[..., 2])

    # Jacobian is a 3x3. Assembling in row-major form
    vals = []
    vals.append(pt[..., 1]*cA*cB - pt[..., 0]*cB*sA)
    vals.append(pt[..., 2]*cB - pt[..., 0]*cA*sB - pt[..., 1]*sA*sB)
    vals.append(0)

    vals.append(- pt[..., 0]*(cA*cG - sA*sB*sG) - pt[..., 1]*(cG*sA + cA*sB*sG))
    vals.append(- pt[..., 2]*sB*sG - pt[..., 0]*cA*cB*sG - pt[..., 1]*cB*sA*sG)
    vals.append(pt[..., 0]*(sA*sG - cA*cG*sB) - pt[..., 1]*(cA*sG + cG*sA*sB) + pt[..., 2]*cB*cG)

    vals.append(pt[..., 0]*(cA*sG + cG*sA*sB) + pt[..., 1]*(sA*sG - cA*cG*sB))
    vals.append(- pt[..., 2]*cG*sB - pt[..., 0]*cA*cB*cG - pt[..., 1]*cB*cG*sA)
    vals.append(pt[..., 0]*(cG*sA + cA*sB*sG) - pt[..., 1]*(cA*cG - sA*sB*sG) - pt[..., 2]*cB*sG)

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
    vals.append(0)

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

# the measurements should have shape timesteps x batches x 6. order is x, y, z, yaw, pitch, roll
# prev biases should have shape batches x 3
# covar matrices should be 3x3, except for prev_covar which is batch_size x 17x17 and nn_covar which is time x batch x 6 x 6

def full_ekf_layer(imu_meas, nn_meas, nn_covar, prev_state, prev_covar, gyro_bias_covar, acc_bias_covar, gyro_covar,
                   acc_covar, dt=tf.constant(0.1, dtype=tf.float32)):
    with tf.variable_scope("ekf_layer"):
        x_output = []
        covar_output = []

        Fklist = []
        Fk = []
        pred_states = []
        pred_covar = []

        prev_states = []

        prev_states.append(prev_state)
        covar_output.append(prev_covar)

# state update. Reused variables are calculated
        lift_g_covar = tf.tile(tf.expand_dims(gyro_covar, axis=0), imu_meas.shape[1])
        lift_a_covar = tf.tile(tf.expand_dims(acc_covar, axis=0), imu_meas.shape[1])
        lift_g_bias_covar = tf.tile(tf.expand_dims(gyro_bias_covar, axis=0), imu_meas.shape[1])
        lift_a_bias_covar = tf.tile(tf.expand_dims(acc_bias_covar, axis=0), imu_meas.shape[1])

        g = tf.constant(-9.80665, dtype=tf.float32)

        gfull = tf.Variable([[0], [0], [-9.80665]], trainable=False)

        diRo = tf.Variable([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -dt, 0],
                            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -dt]], trainable=False)

        dbacc = tf.Variable([[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], trainable=False)

        diRim1 = tf.Variable([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -dt, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -dt, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -dt]], trainable=False)

        dbgy = tf.Variable([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], trainable=False)

        # Prepare the constant part of Fk and tile across all batches
        fkstat = tf.tile(tf.expand_dims(tf.concat([diRo, dbacc, diRim1, dbgy], axis=0), axis=0), imu_meas.shape[1])

# hk is mercifully constant
        hk = tf.Variable([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]], trainable=False)

        Hk = tf.tile(tf.expand_dims(hk, axis=0), imu_meas.shape[1])


        for i in range(imu_meas.shape[0]):
            pred_rot_euler = imu_meas[i, ..., 0:3] - dt * prev_states[i][..., 14:17]
            pred_rot = euler2rot(pred_rot_euler)
            pred_global = imu_meas[i, ..., 1:3] - dt * prev_states[i][..., 15:17] + prev_states[i][..., 6:8]

            pred_global_rot = euler2rot2param(pred_global)

            pred_list = []

            # position prediction
            pred_list.append(dt * tf.matmul(pred_rot, prev_states[i][..., 3:6]) + (0.5 * dt * dt) * \
                             (tf.matmul(pred_global_rot[..., :, 2:3], g) + imu_meas[..., 3:6] - prev_states[i][..., 8:11]))
            # velocity prediction
            pred_list.append(tf.matmul(pred_rot, prev_state[..., 3:6]) + dt * \
                             (tf.matmul(pred_global_rot[..., :, 2:3], g) + imu_meas[..., 3:6] - prev_state[..., 8:11]))

            # global pitch and roll prediction
            pred_list.append(pred_global)

            # accelerometer bias prediction
            pred_list.append(prev_states[i][..., 8:11])

            # relative orientation prediction
            pred_list.append(pred_rot_euler)

            # gyro bias update
            pred_list.append(prev_states[i][..., 14:17])

            # pack state
            pred_states.append(tf.reshape(tf.stack(pred_list), [17]))

            # build Jacobians
            ones = tf.ones([3, 1])
            dR_dE = getJacobian(pred_rot_euler, tf.tile(tf.expand_dims(ones, axis=0), imu_meas.shape[1]))

            Fk = tf.concat([tf.tile(tf.expand_dims(tf.zeros([3, 3]), axis=0), imu_meas.shape[1]),
                            dt * pred_rot,
                            dt * g * getLittleJacobian(pred_global),
                            -dt * tf.tile(tf.expand_dims(tf.eye(dim=3), axis=0), imu_meas.shape[1]),
                            tf.tile(tf.expand_dims(tf.zeros([3, 3]), axis=0), imu_meas.shape[1]),
                            getJacobian(pred_rot_euler,
                                        tf.tile(tf.expand_dims(dt * gfull, axis=0), imu_meas.shape[1]))], axis=-1)

            Fkfull = tf.concat([fkstat, Fk], axis=1)

            # Prepare state noise covariance

            dRglobal_dE = tf.concat([tf.zeros([imu_meas.shape[1], 3, 1]), getLittleJacobian(pred_global)], axis=-1)
            covar_g = lift_g_bias_covar + lift_g_covar
            covar_a = lift_a_bias_covar + lift_a_covar

            # First row of the covariance matrix

            sigma_P_P = dt*dt*dt*dt * tf.matmul(dR_dE, tf.matmul(covar_g, tf.transpose(dR_dE))) + \
                dt*dt*dt*dt*dt * 0.5 * tf.matmul(dR_dE, tf.matmul(covar_g, tf.transpose(dRglobal_dE))) + \
                dt*dt*dt*dt*dt * 0.5 * tf.matmul(dRglobal_dE, tf.matmul(covar_g, tf.transpose(dR_dE))) + \
                dt*dt*dt*dt*dt*dt * 0.25 * tf.matmul(dRglobal_dE, tf.matmul(covar_g, tf.transpose(dRglobal_dE))) + \
                dt*dt*dt*dt * 0.25 * covar_a

            sigma_P_V = dt*dt*dt * tf.matmul(dR_dE, tf.matmul(covar_g, tf.transpose(dR_dE))) + \
                dt*dt*dt*dt * tf.matmul(dR_dE, tf.matmul(covar_g, tf.transpose(dRglobal_dE))) + \
                dt*dt*dt*dt * 0.5 * tf.matmul(dRglobal_dE, tf.matmul(covar_g, tf.transpose(dR_dE))) + \
                dt*dt*dt*dt*dt * 0.5 * tf.matmul(dRglobal_dE, tf.matmul(covar_g, tf.transpose(dRglobal_dE))) + \
                dt*dt*dt * 0.5 * covar_a

            sigma_P_iRo = -dt*dt*dt * tf.matmul(dR_dE, covar_g[..., 1:3]) - \
                dt*dt*dt*dt * 0.5 * tf.matmul(dRglobal_dE, covar_g[..., 1:3])

            sigma_P_bacc = dt *dt * 0.5 * lift_a_bias_covar

            sigma_P_iRim1 = -dt *dt *dt * tf.matmul(dR_dE, covar_g) - dt*dt*dt*0.5 * tf.matmul(dRglobal_dE, covar_g)

            sigma_P_bgy = -dt *dt * tf.matmul(dR_dE, lift_g_bias_covar) - dt*dt*0.5 * tf.matmul(dRglobal_dE, lift_g_bias_covar)

            # Second row

            sigma_V_V = dt*dt * tf.matmul(dR_dE, tf.matmul(covar_g, tf.transpose(dR_dE))) + \
                dt*dt*dt * tf.matmul(dR_dE, tf.matmul(covar_g, tf.transpose(dRglobal_dE))) + \
                dt*dt*dt * tf.matmul(dRglobal_dE, tf.matmul(covar_g, tf.transpose(dR_dE))) + \
                dt*dt*dt*dt* tf.matmul(dRglobal_dE, tf.matmul(covar_g, tf.transpose(dRglobal_dE))) + \
                dt*dt * covar_a

            sigma_V_iRo = -dt*dt * tf.matmul(dR_dE, covar_g[..., 1:3]) - \
                dt*dt*dt * tf.matmul(dRglobal_dE, covar_g[..., 1:3])

            sigma_V_bacc = dt * lift_a_bias_covar

            sigma_V_iRim1 = -dt *dt * tf.matmul(dR_dE, covar_g) - dt*dt * tf.matmul(dRglobal_dE, covar_g)

            sigma_V_bgy = -dt * tf.matmul(dR_dE, lift_g_bias_covar) - dt * tf.matmul(dRglobal_dE, lift_g_bias_covar)

            # Third Row

            sigma_iRo_iRo = dt*dt*covar_g

            sigma_iRo_bacc = tf.zeros([imu_meas.shape[1], 2, 3])

            sigma_iRo_iRim1 = dt*dt*covar_g

            sigma_iRo_bgy = dt * lift_g_bias_covar

            # Fourth Row

            sigma_bacc_bacc = lift_a_bias_covar

            sigma_bacc_iRim1 = tf.zeros([imu_meas.shape[1], 3, 3])

            sigma_bacc_bgy = tf.zeros([imu_meas.shape[1], 3, 3])

            # Fifth Row

            sigma_iRim1_iRim1 = dt * dt * covar_g

            sigma_iRim1_bgy = -dt * lift_g_bias_covar

            # Sixth Row

            sigma_bgy_bgy = lift_g_bias_covar

            # Assemble global covariance matrix

            Qk = tf.concat([
                tf.concat([sigma_P_P, sigma_P_V, sigma_P_iRo, sigma_P_bacc, sigma_P_iRim1, sigma_P_bgy], axis=-1),
                tf.concat([tf.transpose(sigma_P_V), sigma_V_V, sigma_V_iRo, sigma_V_bacc, sigma_V_iRim1, sigma_V_bgy], axis=-1),
                tf.concat([tf.transpose(sigma_P_iRo), tf.transpose(sigma_V_iRo), sigma_iRo_iRo, sigma_iRo_bacc, sigma_iRo_iRim1, sigma_iRo_bgy], axis=-1),
                tf.concat([tf.transpose(sigma_P_bacc), tf.transpose(sigma_V_bacc), tf.transpose(sigma_iRo_bacc), sigma_bacc_bacc, sigma_bacc_iRim1, sigma_bacc_bgy], axis=-1),
                tf.concat([tf.transpose(sigma_P_iRim1), tf.transpose(sigma_V_iRim1), tf.transpose(sigma_iRo_iRim1), tf.transpose(sigma_bacc_iRim1), sigma_iRim1_iRim1, sigma_iRim1_bgy], axis=-1),
                tf.concat([tf.transpose(sigma_P_bgy), tf.transpose(sigma_V_bgy), tf.transpose(sigma_iRo_bgy), tf.transpose(sigma_bacc_bgy), tf.transpose(sigma_iRim1_bgy), sigma_bgy_bgy], axis=-1)],
                axis=-2)

            pred_covar.append(tf.matmul(Fkfull, tf.matmul(covar_output[i], tf.transpose(Fkfull))) + Qk)

            yk = nn_meas[i, ...] - tf.matmul(Hk, pred_states[i])
            Sk = tf.matmul(Hk, tf.matmul(pred_covar[i], tf.transpose(Hk))) + nn_covar[i, ...]
            Kk = tf.matmul(pred_covar[i], tf.matmul(tf.transpose(Hk), tf.matrix_inverse(Sk)))

            x_output.append(pred_states[i] + tf.matmul(Kk, yk))
            covar_output.append(pred_covar[i] - tf.matmul(Kk, tf.matmul(Hk, pred_covar[i])))

    return tf.stack(x_output), tf.stack(covar_output)