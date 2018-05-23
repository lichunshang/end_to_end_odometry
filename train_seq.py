import train
import config

trainer = train.Train(num_gpu=2,
                      cfg=config.SeqTrainLidarConfig,
                      train_sequences=["00", "01", "02", "08", "09"],
                      val_sequence="07",
                      start_epoch=0,
                      restore_file="/home/cs4li/Dev/end_to_end_visual_odometry/results/train_seq_20180521-14-26-32_no_interp_no_init_ts4/best_val/model_best_val_checkpoint-184")

trainer.train()

