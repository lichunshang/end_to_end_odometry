from __future__ import absolute_import, division, print_function
import config

cfg = config.SeqTrainLidarConfig
import train

trainer = train.Train(num_gpu=1,
                      cfg=cfg,
                      #train_sequences=["00", "01", "02", "08", "09"],
                      train_sequences=["04"],
                      val_sequence="05",
                      start_epoch=0,
                      restore_file='/media/cs4li/DATADisk/train_seq_20180528-10-34-14/best_val/model_best_val_checkpoint-195')

trainer.train()
