from tensorflow.python.tools import inspect_checkpoint as chkp

restore_model_file = "/home/cs4li/Dev/end_to_end_odometry/results/train_seq_20180522-14-03-38/model_epoch_checkpoint-199"

chkp.print_tensors_in_checkpoint_file(restore_model_file, tensor_name='',
                                      all_tensors=False, all_tensor_names=True)
