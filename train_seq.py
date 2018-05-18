import train
import config

trainer = train.Train(num_gpu=2,
                      cfg=config.SeqTrainLidarConfig,
                      train_sequences=["00", "01", "02", "08", "09"],
                      val_sequence="07",
                      start_epoch=0,
                      restore_file="/home/cs4li/Dev/end_to_end_visual_odometry/results/train_seq_20180514-10-48-40_no_interp_stateinitlength1/model_epoch_checkpoint-199")

trainer.train()

