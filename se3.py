import tensorflow as tf
import math


# given input tensors of shape containing normalized quaternions
# q1 [:,:,4]
# q2 [:,:,4]

# returns
# [:,:,4] the angular difference between each rotation
def quat_multiply_norm(q1, q2):
    with tf.variable_scope("quat_multiply"):
        w1, x1, y1, z1 = tf.unstack(q1, axis=-1, num=4)
        w2, x2, y2, z2 = tf.unstack(q2, axis=-1, num=4)

        w = tf.multiply(w1, w2) - tf.multiply(x1, x2) - tf.multiply(y1, y2) - tf.multiply(z1, z2)
        x = tf.multiply(w1, x2) + tf.multiply(x1, w2) + tf.multiply(y1, z2) - tf.multiply(z1, y2)
        y = tf.multiply(w1, y2) + tf.multiply(y1, w2) + tf.multiply(z1, x2) - tf.multiply(x1, z2)
        z = tf.multiply(w1, z2) + tf.multiply(z1, w2) + tf.multiply(x1, y2) - tf.multiply(y1, x2)

        return tf.stack((w, x, y, z), axis=-1)


def quat_conjugate(q):
    with tf.variable_scope("quat_conjugate"):
        w, x, y, z = tf.unstack(q, axis=-1, num=4)

        return tf.stack((w, -x, -y, -z), axis=-1)


# given input tensors of shape containing normalized quaternions
# q1 [:,:,4]
# q2 [:,:,4]

# returns
# [:,:,1] the angular difference between each rotation
def quat_subtract(q1, q2):
    with tf.variable_scope("quat_subtract"):
        diff = quat_multiply_norm(q2, quat_conjugate(q1))

        #need to wrap angles because acos produces 0 to 2pi

        ang = tf.multiply(tf.constant(2, dtype=tf.float32), tf.acos(diff[:,:,0]))

        gtcond = tf.greater(ang, tf.constant(math.pi, dtype=tf.float32))

        return tf.where(gtcond, tf.subtract(ang, tf.constant(2*math.pi)), ang)

# input pose_ypr = [x, y, z, yaw, pitch, roll]
# output pose_quat = [x, y, z, qr, qx, qy, qz]
def pose_ypr_to_quat(pose_ypr):
    with tf.variable_scope("pose_ypr_to_quat"):
        p = pose_ypr
        half_y = p[3] / 2.0
        half_p = p[4] / 2.0
        half_r = p[5] / 2.0

        cy = tf.cos(half_y)
        cp = tf.cos(half_p)
        cr = tf.cos(half_r)
        sy = tf.sin(half_y)
        sp = tf.sin(half_p)
        sr = tf.sin(half_r)

        qr = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        n = tf.sqrt(qr ** 2 + qx ** 2 + qy ** 2 + qz ** 2)
        return tf.stack([pose_ypr[0], pose_ypr[1], pose_ypr[2], qr / n, qx / n, qy / n, qz / n])


# input p = [x, y, z, qr, qx, qy, qz], a = [x y z]
# output a_rotated = [x y z]
def comp_pose_pt(p, a):
    with tf.variable_scope("comp_pose_pt"):
        ax, ay, az = tf.unstack(a)
        x, y, z, qr, qx, qy, qz = tf.unstack(p)

        ar_x = x + ax + 2 * (-(qy ** 2 + qz ** 2) * ax + (qx * qy - qr * qz) * ay + (qr * qy + qx * qz) * az)
        ar_y = y + ay + 2 * ((qr * qz + qx * qy) * ax - (qx ** 2 + qz ** 2) * ay + (qy * qz - qr * qx) * az)
        ar_z = z + az + 2 * ((qx * qz - qr * qy) * ax + (qr * qx + qy * qz) * ay - (qx ** 2 + qy ** 2) * az)

        return tf.stack([ar_x, ar_y, ar_z])


# input: q1 = [qr qx qy qz], q2 = [qr qx qy qz]
# output: q = [qr qx qy qz]
def quat_multiply(q1, q2):
    with tf.variable_scope("quat_multiply"):
        r1, x1, y1, z1 = tf.unstack(q1)
        r2, x2, y2, z2 = tf.unstack(q2)
        r = r1 * r2 - x1 * x2 - y1 * y2 - z1 * z2
        x = r1 * x2 + x1 * r2 + y1 * z2 - z1 * y2
        y = r1 * y2 + y1 * r2 + z1 * x2 - x1 * z2
        z = r1 * z2 + z1 * r2 + x1 * y2 - y1 * x2
        n = tf.sqrt(r ** 2 + x ** 2 + y ** 2 + z ** 2)
        return tf.stack([r, x, y, z]) / n


# input: p1 = [x y z qr qx qy qz], p2 = [x y z yaw pitch roll
# output: p = [x y z qr qx qy qz]
def se3_comp(pose_1_quat, pose_2_ypr):
    with tf.variable_scope("se3_comp"):
        # first convert ypr to quaternion
        pose_2_quat = pose_ypr_to_quat(pose_2_ypr)

        x, y, z = tf.unstack(comp_pose_pt(pose_1_quat, pose_2_quat[0:3]))
        qr, qx, qy, qz = tf.unstack(quat_multiply(pose_1_quat[3:7], pose_2_quat[3:7]))

        return tf.stack([x, y, z, qr, qx, qy, qz])
