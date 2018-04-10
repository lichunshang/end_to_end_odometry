from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file(
    "/home/cs4li/Dev/end_to_end_visual_odometry/results/flownet_weights/flownet_s_weights", tensor_name='',
    all_tensors=True, all_tensor_names=True)
