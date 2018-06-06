import train
import config

trainer = train.Train(num_gpu=1,
                      cfg=config.SeqTrainLidarConfig,
                      #train_sequences=["00", "01", "02", "08", "09"],
                      train_sequences=["04"],
                      val_sequence="05",
                      start_epoch=0,
                      restore_file=None)

trainer.train()

