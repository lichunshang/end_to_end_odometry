import tensorflow as tf
import se3
import math as m


def se3_losses_all_ts(outputs, labels, k):
    with tf.variable_scope("se3_losses"):
        diff_p = outputs[:, :, 0:3] - labels[:, :, 0:3]
        # diff_q = outputs[:, :, 3:] - labels[:, :, 3:]
        q_dot_squared = tf.square(tf.reduce_sum(tf.multiply(outputs[:, :, 3:], labels[:, :, 3:]), 2))
        diff_q = tf.subtract(tf.constant(1.0, dtype=tf.float32), q_dot_squared)

        # takes the the dot product and sum it up along time
        sum_diff_p_dot_p = tf.reduce_sum(tf.multiply(diff_p, diff_p), axis=(0, 2,))
        # sum_diff_q_dot_q = tf.reduce_sum(tf.multiply(diff_q, diff_q), axis=(0, 2,))
        sum_diff_q_dot_q = tf.reduce_sum(diff_q, 0)

        t = tf.cast(tf.shape(outputs)[0], tf.float32)

        # multiplies the sum by 1 / t
        loss = (sum_diff_p_dot_p + k * sum_diff_q_dot_q) / t

        return tf.reduce_mean(loss), tf.reduce_mean(sum_diff_p_dot_p / t), tf.reduce_mean(sum_diff_q_dot_q / t)

# assumes time major
def se3_losses_last_ts(outputs, labels, k):
    with tf.variable_scope("se3_losses"):
        diff_p = outputs[-1, :, 0:3] - labels[-1, :, 0:3]
        # diff_p = outputs[:, :, 0:3] - labels[:, :, 0:3]

        q_dot_squared = tf.square(tf.reduce_sum(tf.multiply(outputs[-1, :, 3:], labels[-1, :, 3:]), 1))
        #q_dot_squared = tf.square(tf.reduce_sum(tf.multiply(outputs[:, :, 3:], labels[:, :, 3:]), 2))
        # diff_q = outputs[:, :, 3:] - labels[:, :, 3:]

        diff_q = tf.subtract(tf.constant(1.0, dtype=tf.float32), q_dot_squared)

        sum_diff_p_dot_p = tf.reduce_sum(tf.multiply(diff_p, diff_p), axis=1)
        # takes the the dot product and sum it up along time
        # sum_diff_p_dot_p = tf.reduce_sum(tf.multiply(diff_p, diff_p), axis=(0, 2,))

        sum_diff_q_dot_q = diff_q
        # sum_diff_q_dot_q = tf.reduce_sum(tf.multiply(diff_q, diff_q), axis=(0, 2,))
        # sum_diff_q_dot_q = tf.reduce_sum(diff_q, 0)

        t = tf.cast(tf.shape(outputs)[0], tf.float32)

        # multiplies the sum by 1 / t
        loss = (sum_diff_p_dot_p + k * sum_diff_q_dot_q) / t

        return tf.reduce_mean(loss), tf.reduce_mean(sum_diff_p_dot_p / t), tf.reduce_mean(sum_diff_q_dot_q / t)

def se3_losses(outputs, labels, k):
    return se3_losses_all_ts(outputs, labels, k)

def pair_train_fc_losses(outputs, labels_u, k):
    with tf.variable_scope("pair_train_fc_losses"):
        diff_p = outputs[:, :, 0:3] - labels_u[:, :, 0:3]
        diff_e = outputs[:, :, 3:6] - labels_u[:, :, 3:6]

        # takes the the dot product and sum it up along time
        diff_p_sq = tf.multiply(diff_p, diff_p)
        sum_diff_p_dot_p = tf.reduce_sum(diff_p_sq, axis=(0, 2,))
        sum_diff_e_dot_e = tf.reduce_sum(tf.multiply(diff_e, diff_e), axis=(0, 2,))

        t = tf.cast(tf.shape(outputs)[0], tf.float32)

        # multiplies the sum by 1 / t
        loss = (sum_diff_p_dot_p + k * sum_diff_e_dot_e) / t

        # return xyz losses
        x_loss = tf.reduce_mean(tf.sqrt(diff_p_sq[:, :, 0]))
        y_loss = tf.reduce_mean(tf.sqrt(diff_p_sq[:, :, 1]))
        z_loss = tf.reduce_mean(tf.sqrt(diff_p_sq[:, :, 2]))

        return tf.reduce_mean(loss), tf.reduce_mean(sum_diff_p_dot_p / t), tf.reduce_mean(sum_diff_e_dot_e / t), \
               x_loss, y_loss, z_loss


# reduce_prod for tensor length 6, x shape is [time length, batch size, 6]
def reduce_prod_6(x):
    r = tf.multiply(x[:, :, 0], x[:, :, 1])
    r = tf.multiply(r, x[:, :, 2])
    r = tf.multiply(r, x[:, :, 3])
    r = tf.multiply(r, x[:, :, 4])
    r = tf.multiply(r, x[:, :, 5])

    return r


# assumes time major
def fc_losses_covar(outputs, output_covar, labels_u, k):
    # return pair_train_fc_losses(outputs, labels_u, k)
    with tf.variable_scope("fc_losses"):
        diff_u = tf.subtract(outputs[:, :, 0:6], labels_u, name="diff_u")
        diff_u2 = tf.square(diff_u)

        # dense covariance
        Q = tf.identity(output_covar, name="Q_check")

        log_det_Q = tf.log(tf.matrix_determinant(Q) + 1e-7)

        # Need to normalize
        norm1 = tf.tile(tf.expand_dims(tf.diag(1e-5*tf.ones([6], dtype=tf.float32)), axis=0), [Q.shape[1], 1, 1])
        norm2 = tf.tile(tf.expand_dims(norm1, axis=0), [Q.shape[0], 1, 1, 1])
        inv_Q = tf.matrix_inverse(Q + norm2)

        # sum of determinants along the time
        sum_det_Q = tf.reduce_sum(log_det_Q, axis=0)

        # need to scale angular error by k
        ksq = tf.sqrt(k)
        diff_u_scaled = tf.concat([diff_u[..., 0:3], ksq * diff_u[..., 3:6]], axis=-1)
        # diff_u_normalized = tf.div(diff_u_scaled, labels_u)

        # sum of diff_u' * inv_Q * diff_u
        s = tf.reduce_sum(tf.squeeze(tf.matmul(tf.expand_dims(diff_u_scaled, axis=-1), tf.matmul(inv_Q, tf.expand_dims(diff_u_scaled, axis=-1)), transpose_a=True), axis=-1), axis=0)

        t = tf.cast(tf.shape(outputs)[0], tf.float32)

        # add and multiplies of sum by 1 / t
        loss = (s + sum_det_Q) / t

        xloss = tf.sqrt(tf.reduce_mean(tf.reduce_sum(diff_u2[..., 0], axis=0), axis=0), name="x_loss_sqrt")
        yloss = tf.sqrt(tf.reduce_mean(tf.reduce_sum(diff_u2[..., 1], axis=0), axis=0))
        zloss = tf.sqrt(tf.reduce_mean(tf.reduce_sum(diff_u2[..., 2], axis=0), axis=0))
        xyzloss = tf.reduce_mean(tf.reduce_sum(diff_u2[..., 0:3], axis=[0, 2]), axis=0)
        yprloss = tf.reduce_mean(tf.reduce_sum(diff_u2[..., 3:6], axis=[0, 2]), axis=0)
        
        mean = tf.reduce_mean(loss, name="reduce_mean_loss")

        return mean, xyzloss, yprloss, xloss, yloss, zloss

def fc_losses(outputs, output_covar, labels_u, k):
    return pair_train_fc_losses(outputs, labels_u, k)
    # return fc_losses_covar(outputs, output_covar, labels_u, k)
