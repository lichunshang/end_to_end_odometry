from __future__ import absolute_import, division, print_function
import config
import train

cfg = config.SeqTrainCamConfig

trainer = train.Train(num_gpu=2,
                      cfg=cfg,
                      train_sequences=["00", "01", "02", "04", "05", "06", "07"],
                      val_sequence="08",
                      start_epoch=0,
                      restore_file=None,
                      restore_ekf_state_file=None)

trainer.train()
