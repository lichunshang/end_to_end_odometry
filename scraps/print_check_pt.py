from tensorflow.python.tools import inspect_checkpoint as chkp

chkp.print_tensors_in_checkpoint_file("/media/bapskiko/Trailing 24GB/model_epoch_checkpoint-140", tensor_name='',
    all_tensors=False, all_tensor_names=True)
