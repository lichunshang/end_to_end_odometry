from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file("/home/cs4li/Dev/end_to_end_visual_odometry/results/train_seq_20180411-19-34-59/best_val/model_best_val_checkpoint-7", tensor_name='',
    all_tensors=False, all_tensor_names=True)
