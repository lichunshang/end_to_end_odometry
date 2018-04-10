from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file(
    "/home/cs4li/Dev/end_to_end_visual_odometry/results/" \
    "train_pair_20180409-23-44-54_75_epochs_0.125_loss/" \
    "model_epoch_checkpoint-74", tensor_name='',
    all_tensors=True, all_tensor_names=True)
