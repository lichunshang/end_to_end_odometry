import data_roller
import config

cfg = config.SeqTrainConfigsSmallSteps
data_roller.StatefulRollerDataGen(cfg, "/home/cs4li/Dev/KITTI/dataset/", ["06"])
