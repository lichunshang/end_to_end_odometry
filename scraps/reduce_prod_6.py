import losses
import numpy as np
import tensorflow as tf

a = np.random.random([151, 163, 6])
b = tf.constant(a, dtype=tf.float32)
sess = tf.Session()

c = sess.run(tf.reduce_prod(b, axis=2))
d = sess.run(losses.reduce_prod_6(b))

print(np.max(c-d))

