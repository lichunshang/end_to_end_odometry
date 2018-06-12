from tensorflow.python.tools import inspect_checkpoint as chkp

restore_model_file = "/media/cs4li/DATADisk/train_seq_20180528-10-34-14/best_val/model_best_val_checkpoint-195"

chkp.print_tensors_in_checkpoint_file(restore_model_file, tensor_name='',
                                      all_tensors=False, all_tensor_names=True)
