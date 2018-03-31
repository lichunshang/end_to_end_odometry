save_path = "/home/lichunshang/Dev/end_to_end_visual_odometry/results"

class TrainConfigs(object):
    timesteps = 0
    batch_size = 0
    input_width = 0
    input_height = 0
    input_channels = 0

class SeqTrainConfigs(TrainConfigs):
    timesteps = 8
    batch_size = 7
    input_width = 1280
    input_height = 384
    input_channels = 1

    lstm_size = 256
    lstm_layers = 2

    num_epochs = 3
    k = 10


class PairTrainConfigs(TrainConfigs):
    timesteps = 1
    batch_size = 50

    input_width = 1280
    input_height = 384
    input_channels = 1

    num_epochs = 3
    k = 1
