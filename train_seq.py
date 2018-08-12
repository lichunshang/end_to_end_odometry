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
                      # restore_file="/home/cs4li/Dev/end_to_end_odometry/results/train_seq_20180808-00-45-36_normalized_error_good_covar/model_epoch_checkpoint-99",
                      # restore_ekf_state_file=None,
                      # restore_ekf_state_file="/media/cs4li/DATADisk/results/train_seq_20180728-00-49-48_16ts_golden_ekf_state/model_epoch_ekf_states-199"
                      restore_file="/media/cs4li/DATADisk/results/train_seq_20180709-10-44-35_4ts_golden/model_epoch_checkpoint-199",
                      restore_ekf_state_file="/media/cs4li/DATADisk/results/train_seq_20180728-00-49-48_16ts_golden_ekf_state/model_epoch_ekf_states-199"
                      )

trainer.train()
