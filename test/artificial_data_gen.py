import numpy as np
import transformations

dt = 0.1
timesteps = 100

curr_p = np.array([0, 0, 0])
curr_v = np.array([0, 0, 0])
curr_q = np.array([1, 0, 0, 0])
a = np.array([0, 0, 0])
w = np.array([0, 0, 0])


def q_v(v):
    phi = np.sqrt(v.dot(v))

    if np.abs(phi) > 1e-15:
        u = v / phi
        # return = [cos(phi / 2) u * sin(phi / 2)];
        return np.concatenate([[np.cos(phi / 2)], u * np.sin(phi / 2)])
    else:
        return np.array([1, 0, 0, 0])


for i in range(0, timesteps):
    R_q = transformations.quaternion_matrix(curr_q)[0:3, 0:3]

    new_p = curr_p + curr_v * dt + 0.5 * R_q * a * dt ** 2
    new_v = curr_v + R_q * a * dt
    new_q = transformations.quaternion_multiply(curr_q, q_v())
