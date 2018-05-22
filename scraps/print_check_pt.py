from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file("/home/cs4li/Dev/end_to_end_visual_odometry/results/train_seq_20180521-14-26-32_no_interp_no_init_ts4/best_val/model_best_val_checkpoint-184", tensor_name='',
    all_tensors=False, all_tensor_names=True)
