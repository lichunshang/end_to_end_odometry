from tensorflow.python.tools import inspect_checkpoint as chkp

restore_model_file = "/home/cs4li/Dev/end_to_end_odometry/results/train_seq_20180622-14-57-25_baseline_converged_with_may28_init/model_epoch_checkpoint-199"

chkp.print_tensors_in_checkpoint_file(restore_model_file, tensor_name='',
                                      all_tensors=False, all_tensor_names=True)
