import tools

save_path = "/home/cs4li/Dev/end_to_end_visual_odometry/results"


class Configs(object):
    timesteps = 0
    batch_size = 0
    input_width = 0
    input_height = 0
    input_channels = 0


class SeqTrainConfigs(Configs):
    timesteps = 4
    batch_size = 7
    input_width = 1280
    input_height = 384
    input_channels = 1

    lstm_size = 256
    lstm_layers = 2

    num_epochs = 100
    k_fc = 1
    k_se3 = 1


class PairTrainConfigs(Configs):
    timesteps = 1
    batch_size = 30

    input_width = 1280
    input_height = 384
    input_channels = 1

    num_epochs = 50
    k = 1


def print_configs(cfg):
    for attr in dir(cfg):
        if not callable(getattr(cfg, attr)) and not attr.startswith("__"):
            tools.printf("%s: %s" % (attr, getattr(cfg, attr)))
