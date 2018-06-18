from __future__ import absolute_import, division, print_function
import config

cfg = config.SeqTrainLidarConfig
import train

trainer = train.Train(num_gpu=2,
                      cfg=cfg,
                      train_sequences=["00", "01", "02", "08", "09"],
                      #train_sequences=["04"],
                      val_sequence="05",
                      start_epoch=0,
                      restore_file=None)

trainer.train()
