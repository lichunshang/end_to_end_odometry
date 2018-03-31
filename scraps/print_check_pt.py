from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file(
    "/home/lichunshang/Dev/end_to_end_visual_odometry/results/train_seq_20180330-21-53-52/saved_model", tensor_name='',
    all_tensors=True, all_tensor_names=True)
