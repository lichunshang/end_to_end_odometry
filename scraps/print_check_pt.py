from tensorflow.python.tools import inspect_checkpoint as chkp

restore_model_file = "/home/cs4li/Dev/end_to_end_odometry/results/train_seq_20180813-12-32-50/model_epoch_checkpoint-145"

chkp.print_tensors_in_checkpoint_file(restore_model_file, tensor_name='',
                                      all_tensors=True, all_tensor_names=True)
