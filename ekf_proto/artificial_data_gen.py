import numpy as np
import transformations
import test.artifical_data_profiles as profiles


def q_v(v):
    phi = np.sqrt(v.dot(v))

    if np.abs(phi) > 1e-15:
        u = v / phi
        return np.concatenate([[np.cos(phi / 2)], u * np.sin(phi / 2)])
    else:
        return np.array([1, 0, 0, 0])


def skew(v):
    m = np.zeros([3, 3])
    m[0, 1] = -v[2]
    m[0, 2] = v[1]
    m[1, 0] = v[2]

    m[1, 2] = -v[0]
    m[2, 0] = -v[1]
    m[2, 1] = v[0]

    return m


profile = profiles.StraightRotate

g = 9.80665
curr_p = profile.init_p
curr_v = profile.init_v
curr_q = profile.init_q

init_file_str = "lat lon alt roll pitch yaw vn ve vf vl vu ax ay az af al au wx wy wz wf wl wu " \
                "pos_accuracy vel_accuracy navstat numsats posmode velmode orimode\n"
init_file_val = np.zeros(30)
init_file_val[8:11] = curr_v
init_file_val[3:6] = transformations.euler_from_quaternion(curr_q, 'rzyx')
init_file_str += " ".join(["%.15f" % x for x in init_file_val])

dat_file_str = "dt dx dy dz dyaw dpitch droll wx wy wz ax ay az gx gy gz gqw gqx gqy gqz\n"

for i in range(0, profile.timesteps):
    R_q = transformations.quaternion_matrix(curr_q)[0:3, 0:3]

    w, a = profile.excitation_policy(i)
    dt = profile.dt

    # print(a)

    new_p = curr_p + curr_v * dt + 0.5 * R_q.dot(a) * dt ** 2
    new_v = curr_v + R_q.dot(a) * dt
    new_q = transformations.quaternion_multiply(curr_q, q_v(w * dt))

    new_T = transformations.quaternion_matrix(new_q)
    new_T[0:3, 3] = new_p

    curr_T = transformations.quaternion_matrix(curr_q)
    curr_T[0:3, 3] = curr_p

    diff_T = np.linalg.inv(curr_T).dot(new_T)
    diff_p = diff_T[0:3, 3]
    diff_euler = transformations.euler_from_matrix(diff_T, 'rzyx')

    curr_R = curr_T[0:3, 0:3]
    a_measured = np.transpose(curr_R).dot(curr_R.dot(a) - np.array([0, 0, -g]))

    dat_file_str += "%.8f  %s  %s  %s\n" % \
                    (dt,
                     " ".join("%.8f" % x for x in np.concatenate([diff_p, diff_euler])),
                     " ".join("%.8f" % x for x in np.concatenate([w, a_measured]) + profile.imu_bias(i)),
                     " ".join("%.8f" % x for x in np.concatenate([new_p, new_q])))

    curr_p = new_p
    curr_v = new_v
    curr_q = new_q

init_file = open("/home/cs4li/Dev/end_to_end_odometry/test/data/artificial_%s_init.dat" % profile.name, "w")
dat_file = open("/home/cs4li/Dev/end_to_end_odometry/test/data/artificial_%s.dat" % profile.name, "w")
init_file.write(init_file_str)
dat_file.write(dat_file_str)
init_file.close()
dat_file.close()
