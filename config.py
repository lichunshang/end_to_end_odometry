import tools
import getpass

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
    sequence_stride = 16
    timesteps = 8
    init_length = 1
    batch_size = 16

    input_width = 1152
    input_height = 64
    input_channels = 2

    debug = False

    bidir_aug = True

    use_init = False
    use_ekf = False

    only_train_init = False  # only used when use init is True
    dont_restore_init = True  # only used when use init is True
    static_nn = False # don't modify the nn weights
    train_noise_covariance = False # Train the imu noise covariance
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

    lr_schedule = {0: 0.0001,
                   20: 0.00005,
                   40: 0.000002,
                   80: 0.000001,
                   130: 0.0000001}


def print_configs(cfg):
    for attr in dir(cfg):
        if not callable(getattr(cfg, attr)) and not attr.startswith("__"):
            tools.printf("%s: %s" % (attr, getattr(cfg, attr)))
