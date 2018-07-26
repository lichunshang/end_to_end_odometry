import tools
import getpass
import numpy as np

machine = getpass.getuser()

if machine == "cs4li":
    save_path = "/home/cs4li/Dev/end_to_end_odometry/results/"
    dataset_path = "/home/cs4li/Dev/KITTI/dataset/"
    lidar_pickles_path = "/home/cs4li/Dev/KITTI/dataset/sequences/lidar_pickles_no_interp"
elif machine == "bapskiko":
    save_path = "/home/bapskiko/git/end_to_end_visual_odometry/results"
    dataset_path = "/media/bapskiko/SpinDrive/kitti/dataset"
    lidar_pickles_path = "/home/bapskiko/git/end_to_end_visual_odometry/pickles"


class Configs(object):
    timesteps = 0
    batch_size = 0
    input_width = 0
    input_height = 0
    input_channels = 0


class SeqTrainConfigs(Configs):
    timesteps = 8
    batch_size = 4
    input_width = 1280
    input_height = 384
    input_channels = 3

    lstm_size = 256
    lstm_layers = 2
    sequence_stride = 8

    num_epochs = 100
    k_fc = 50
    k_se3 = 500


class SeqTrainLidarConfig(Configs):
    timesteps = 32
    sequence_stride = timesteps
    init_length = 1
    batch_size = 4

    input_width = 1152
    input_height = 64
    input_channels = 2

    bidir_aug = True  # train going in reverse as well

    # EKF stuff
    use_ekf = True
    train_noise_covariance = True  # Train the imu noise covariance for ekf
    static_nn = True  # don't modify the nn weights if set to true
    fix_fc_covar = True
    fc_covar_fix_val = np.array([0.1] * 6, dtype=np.float32)
    ekf_initial_state_covariance = 100  # initial covariance for all the states
    init_gyro_bias_covar = 10
    init_acc_bias_covar = 10
    init_gyro_covar = 10
    init_acc_covar = 10

    train_ekf_with_fcgt = True  # train ekf using fc ground truth instead of nn outputs
    gt_init_vel_state = True  # use ground truth for initial velocity state for all batches at every epoch

    # initializer stuff
    use_init = False
    only_train_init = False  # only used when use init is True
    dont_restore_init = True  # only used when use init is True

    debug = False

    init_prob = 1

    data_type = "lidar"

    lstm_size = 256
    lstm_layers = 1

    k_fc = 50.0
    k_se3 = 500.0

    num_epochs = 200
    alpha_schedule = {0: 0.8}

    # lr_schedule = {0: 0.000002,
    #                50: 0.000001,
    #                100: 0.0000001}

    lr_schedule = {0: 0.01,
                   20: 0.005,
                   40: 0.0002,
                   80: 0.0001,
                   130: 0.00001}


def print_configs(cfg):
    for attr in dir(cfg):
        if not callable(getattr(cfg, attr)) and not attr.startswith("__"):
            tools.printf("%s: %s" % (attr, getattr(cfg, attr)))
