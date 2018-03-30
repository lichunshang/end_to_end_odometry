save_path = "/home/lichunshang/Dev/end_to_end_visual_odometry/results"

class TrainConfigs(object):
    timesteps = -1
    batch_size = -1
    input_width = -1
    input_height = -1
    input_channels = -1

class SeqTrainConfigs:
    timesteps = 8
    batch_size = 7
    input_width = 1280
    input_height = 384
    input_channels = 1

    lstm_size = 256
    lstm_layers = 2

    num_epochs = 3
    k = 10


class PairTrainConfigs:
    timesteps = 1
    batch_size = 1

    input_width = 1280
    input_height = 384
    input_channels = 1

    num_epochs = 3
    k = 1
