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
def se3_losses(imu_meas, nn_meas, nn_covar, prev_bias, prev_covar, bias_covar, imu_covar, timestep=tf.constant(0.1, dtype=tf.float32)):
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

# todo version that incorporates linear acceleration. Bit more because need to know orientation relative to gravity