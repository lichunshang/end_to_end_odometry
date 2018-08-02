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
                      restore_file="/media/cs4li/DATADisk/results/train_seq_20180709-10-44-35_4ts_golden/model_epoch_checkpoint-199",
                      restore_ekf_state_file=None)

trainer.train()
