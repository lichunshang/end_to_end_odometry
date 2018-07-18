from __future__ import absolute_import, division, print_function
import config
import train

cfg = config.SeqTrainLidarConfig

trainer = train.Train(num_gpu=2,
                      cfg=cfg,
                      train_sequences=["00", "01", "02", "08", "09"],
                      # train_sequences=["04"],
                      val_sequence="07",
                      start_epoch=0,
                      restore_file="/home/cs4li/Dev/end_to_end_odometry/results/train_seq_20180716-17-55-07_32ts_golden/model_epoch_checkpoint-199")

trainer.train()
