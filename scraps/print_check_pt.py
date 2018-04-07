from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file(
    "/home/cs4li/Dev/end_to_end_visual_odometry/results/train_seq_20180406-17-59-50_seq_all_cnn_init_cosine_dist/model_epoch_checkpoint-49", tensor_name='',
    all_tensors=True, all_tensor_names=True)
