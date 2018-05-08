import train
import config

trainer = train.Train(num_gpu = 2,
                      config = config.SeqTrainLidarConfig,
                      train_sequences=["00", "01", "02", "08", "09"],
                      val_sequence="07",
                      start_epoch=0,
                      restore_file=None)

trainer.train()

