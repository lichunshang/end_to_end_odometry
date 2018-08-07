from tensorflow.python.debug.lib.debug_data import InconvertibleTensorProto

def blowup_filter(datum, tensor):
  # A filter that detects zero-valued scalars.
  # return "x_loss_sqrt" in tensor.name and tensor > 1.0
  # return len(tensor.shape) == 0 and tensor == 0.0
  # return tensor > 1.0
  # return len(tensor.shape) == 0 and tensor == 0.0

  if isinstance(tensor, InconvertibleTensorProto):
    # Uninitialized tensor doesn't have bad numerical values.
    # Also return False for data types that cannot be represented as numpy
    # arrays.
    return False
  elif "x_loss_sqrt" in datum.node_name and len(tensor.shape) == 0 and tensor >= 2.0:
  	return True
  else:
    return False
