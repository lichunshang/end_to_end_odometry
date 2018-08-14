from __future__ import absolute_import, division, print_function
import config
import train

cfg = config.SeqTrainLidarConfig

trainer = train.Train(num_gpu=2,
                      cfg=cfg,
                      train_sequences=["00", "01", "02", "04","05", "06", "07"],
                      # train_sequences=["04"],
                      val_sequence="08",
                      start_epoch=0,
                      restore_file=None,
                      # restore_file="/media/cs4li/DATA/end_to_end_odometry/results/train_seq_20180812-16-22-19/model_epoch_checkpoint-199",
                      restore_ekf_state_file=None,
                      # restore_ekf_state_file="/media/cs4li/DATADisk/results/train_seq_20180728-00-49-48_16ts_golden_ekf_state/model_epoch_ekf_states-199"
                      )

trainer.train()
