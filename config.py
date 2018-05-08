import tools

machine = "cs4li"

if machine == "cs4li":
    save_path = "/home/cs4li/Dev/end_to_end_visual_odometry/results/"
    dataset_path = "/home/cs4li/Dev/KITTI/dataset/"
    lidar_pickles_path = "/home/cs4li/Dev/KITTI/dataset/sequences/lidar_pickles/"


class Configs(object):
    timesteps = 0
    batch_size = 0
    input_width = 0
    input_height = 0
    input_channels = 0

    def print_configs(self):
        for attr in dir(self):
            if not callable(getattr(self, attr)) and not attr.startswith("__"):
                tools.printf("%s: %s" % (attr, getattr(self, attr)))


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
    sequence_stride = 1
    timesteps = 1
    batch_size = 24
    input_width = 1152
    input_height = 64
    input_channels = 3

    bidir_aug = True
    data_type = "lidar"

    lstm_size = 256
    lstm_layers = 1

    k_fc = 50
    k_se3 = 500

    num_epochs = 200
    alpha_schedule = {0: 0.99,
                      20: 0.9,
                      40: 0.5,
                      60: 0.1,
                      80: 0.025}

    lr_schedule = {0: 0.0001,
                   3: 0.00008,
                   7: 0.00005,
                   13: 0.000002,
                   20: 0.000001,
                   50: 0.0000001}


class SeqEvalLidarConfig(Configs):
    timesteps = 1
    batch_size = 1
    input_width = 1152
    input_height = 64
    input_channels = 3

    bidir_aug = False
    data_type = "lidar"

    lstm_size = 256
    lstm_layers = 1
    sequence_stride = 1

    num_epochs = 200
    k_fc = 50
    k_se3 = 500


class SeqCamEvalConfig(Configs):
    timesteps = 1
    batch_size = 1
    input_width = 1280
    input_height = 384
    input_channels = 3
    sequence_stride = 1
    bidir_aug = False

    lstm_size = 256
    lstm_layers = 2
