import numpy as np


class StraightProfile(object):
    name = "default"
    dt = 0.1
    timesteps = 100
    init_p = np.array([0, 0, 0])
    init_v = np.array([1, 0, 0])
    init_q = np.array([1, 0, 0, 0])

    @staticmethod
    def excitation_policy(timestep):
        accel = np.array([0, 0, 0])
        gyro = np.array([0, 0, 0])
        return gyro, accel

    @staticmethod
    def imu_bias(timestep):
        return np.zeros([6])


class StraightProfileWithBias(object):
    name = "default"
    dt = 0.1
    timesteps = 10000
    init_p = np.array([0, 0, 0])
    init_v = np.array([1, 0, 0])
    init_q = np.array([1, 0, 0, 0])

    @staticmethod
    def excitation_policy(timestep):
        accel = np.array([0, 0, 0])
        gyro = np.array([0, 0, 0])
        return gyro, accel

    @staticmethod
    def imu_bias(timestep):
        return np.array([0.0003, -0.002, 0.001, 0.1, -0.02, 0.003])


class SineAccel(object):
    name = "default"
    dt = 0.1
    timesteps = 1000
    init_p = np.array([0, 0, 0])
    init_v = np.array([1, 0, 0])
    init_q = np.array([1, 0, 0, 0])

    @staticmethod
    def excitation_policy(timestep):
        accel = np.array([0, 0, 0], dtype=np.float32)
        gyro = np.array([0, 0, 0], dtype=np.float32)

        accel[0] = np.sin(timestep * SineAccel.dt / 10)
        accel[1] = np.sin(timestep * SineAccel.dt / 10 + np.pi / 6)
        accel[2] = np.sin(timestep * SineAccel.dt / 10 + np.pi / 3)
        return gyro, accel

    @staticmethod
    def imu_bias(timestep):
        return np.zeros([6])


class StraightRotate(object):
    name = "default"
    dt = 0.1
    timesteps = 10000
    init_p = np.array([0, 0, 0])
    init_v = np.array([1, 0, 0])
    init_q = np.array([1, 0, 0, 0])

    @staticmethod
    def excitation_policy(timestep):
        accel = np.array([0, 0, 0], dtype=np.float32)
        gyro = np.array([0, 0, 0], dtype=np.float32)
        gyro[0] = 0.023
        gyro[1] = -0.036
        gyro[2] = 0.029
        return gyro, accel

    @staticmethod
    def imu_bias(timestep):
        return np.zeros([6])



class SineAccelRotating(object):
    name = "default"
    dt = 0.1
    timesteps = 10000
    init_p = np.array([0, 0, 0])
    init_v = np.array([1, 0, 0])
    init_q = np.array([1, 0, 0, 0])

    @staticmethod
    def excitation_policy(timestep):
        accel = np.array([0, 0, 0], dtype=np.float32)
        gyro = np.array([0, 0, 0], dtype=np.float32)

        accel[0] = np.sin(timestep * SineAccel.dt / 10)
        accel[1] = np.sin(timestep * SineAccel.dt / 10 + np.pi / 6)
        accel[2] = np.sin(timestep * SineAccel.dt / 10 + np.pi / 3)

        gyro[0] = 0.01
        gyro[1] = -0.02
        gyro[2] = -0.01

        return gyro, accel

    @staticmethod
    def imu_bias(timestep):
        return np.zeros([6])