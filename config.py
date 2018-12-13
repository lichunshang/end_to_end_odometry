import tools
import getpass
import numpy as np

machine = getpass.getuser()

if machine == "cs4li":
    save_path = "/home/cs4li/Dev/end_to_end_odometry/results/"
    dataset_path = "/home/cs4li/Dev/KITTI/dataset/"

    kitti_dataset_path = "/home/cs4li/Dev/KITTI/dataset/"
    kitti_lidar_pickles_path = "/home/cs4li/Dev/KITTI/dataset/sequences/lidar_pickles_raw_no_interp"


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


class SeqTrainCamConfig(Configs):
    timesteps = 8
    sequence_stride = timesteps
    init_length = 1
    batch_size = 8

    input_width = 1280
    input_height = 384
    input_channels = 3

    bidir_aug = True  # train going in reverse as well

    # EKF stuff
    use_ekf = False
    train_noise_covariance = True  # Train the imu noise covariance for ekf
    fix_fc_covar = False
    fc_covar_fix_val = np.array([0.1] * 6, dtype=np.float32)
    ekf_initial_state_covariance = 100  # initial covariance for all the states
    init_gyro_bias_covar = 10
    init_acc_bias_covar = 10
    init_gyro_covar = 10
    init_acc_covar = 10
    train_ekf_with_fcgt = False  # train ekf using fc ground truth instead of nn outputs
    gt_init_vel_state = True  # use ground truth for initial velocity state for all batches at every epoch
    gt_init_vel_state_only_first = False  # if gt_init_vel_state=True, but only force the first epoch first batch

    static_nn = False  # don't modify the nn weights if set to true

    # initializer stuff
    use_init = False
    only_train_init = False  # only used when use init is True
    dont_restore_init = True
    dont_restore_fc = True
    dont_restore_lstm = True

    debug = False

    init_prob = 1

    data_type = "cam"

    lstm_size = 256
    lstm_layers = 1

    k_fc = 50.0
    k_se3 = 500.0

    num_epochs = 200
    alpha_schedule = {0: 0.25}

    lr_schedule = {0: 0.0001,
                   50: 0.00002,
                   100: 0.000004,
                   135: 0.0000008,
                   170: 0.0000001}


def print_configs(cfg):
    for attr in dir(cfg):
        if not callable(getattr(cfg, attr)) and not attr.startswith("__"):
            tools.printf("%s: %s" % (attr, getattr(cfg, attr)))
