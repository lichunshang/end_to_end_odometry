from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file(
    "/home/cs4li/Dev/end_to_end_visual_odometry/results/train_pair_20180401-23-44-01_seq_00_to_05_not_randomized_dropout/model_checkpoint", tensor_name='',
    all_tensors=True, all_tensor_names=True)
