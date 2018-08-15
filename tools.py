from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import tensor_shape
import tensorflow as tf
import config
import os
import sys
import datetime
import numpy as np
import collections
from shutil import copyfile


# TODO(yuanbyu, mrry): Handle stride to support sliding windows.
def foldl(fn, elems, dtype=None, initializer=None, parallel_iterations=10, back_prop=True,
          swap_memory=False, name=None):
    """foldl that returns results after each iteration instead of just the last
    one. It is the same as the tensorflow foldl otherwise
    """
    if not callable(fn):
        raise TypeError("fn must be callable.")

    if initializer is None:
        raise TypeError("There must be a initializer")

    if dtype is None:
        raise TypeError("There must be a type")

    in_graph_mode = context.in_graph_mode()
    with ops.name_scope(name, "foldl", [elems]):
        # TODO(akshayka): Remove the in_graph_mode check once caching devices are
        # supported in Eager
        if in_graph_mode:
            # Any get_variable calls in fn will cache the first call locally
            # and not issue repeated network I/O requests for each iteration.
            varscope = vs.get_variable_scope()
            varscope_caching_device_was_none = False
            if varscope.caching_device is None:
                # TODO(ebrevdo): Change to using colocate_with here and in other
                # methods.
                varscope.set_caching_device(lambda op: op.device)
                varscope_caching_device_was_none = True

        # Convert elems to tensor array.
        elems = ops.convert_to_tensor(elems, name="elems")
        n = array_ops.shape(elems)[0]
        elems_ta = tensor_array_ops.TensorArray(dtype=elems.dtype, size=n,
                                                dynamic_size=False,
                                                infer_shape=True)
        elems_ta = elems_ta.unstack(elems)

        a = ops.convert_to_tensor(initializer)
        i = constant_op.constant(1)

        ta = tensor_array_ops.TensorArray(dtype=dtype, size=n + 1, dynamic_size=False, infer_shape=True,
                                          clear_after_read=False)
        ta = ta.write(0, a)

        def compute(i, ta):
            a = fn(ta.read(i - 1), elems_ta.read(i - 1))
            ta = ta.write(i, a)
            return [i + 1, ta]

        _, r_a = control_flow_ops.while_loop(
                lambda i, a: i < n + 1, compute, [i, ta],
                parallel_iterations=parallel_iterations,
                back_prop=back_prop,
                swap_memory=swap_memory)

        # TODO(akshayka): Remove the in_graph_mode check once caching devices are
        # supported in Eager
        if in_graph_mode and varscope_caching_device_was_none:
            varscope.set_caching_device(None)

        r_a = r_a.stack()
        r_a = r_a[1:]
        r_a.set_shape(elems.get_shape())
        return r_a


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y


def static_map_fn(fn, inputs, axis):
    unstacked_inputs = tf.unstack(inputs, axis=axis)

    outputs = []

    for val in unstacked_inputs:
        outputs.append(fn(val))

    return tf.stack(outputs, axis=axis)


def create_results_dir(prepend):
    dir_name = prepend + "_%s" % datetime.datetime.today().strftime('%Y%m%d-%H-%M-%S')
    results_dir_path = os.path.join(config.save_path, dir_name)
    if not os.path.exists(results_dir_path):
        os.makedirs(results_dir_path)

    best_val_path = os.path.join(results_dir_path, "best_val")
    if not os.path.exists(best_val_path):
        os.makedirs(best_val_path)

    return results_dir_path


def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def ensure_file_dir_exists(path):
    make_dir_if_not_exist(os.path.dirname(path))
    return path


def log_file_content(log_path, file_paths):
    common_prefix = os.path.commonprefix(file_paths)
    for f in file_paths:
        rel_path = os.path.relpath(f, common_prefix)
        copyfile(f, ensure_file_dir_exists(os.path.join(log_path, "code_log", rel_path)))


file_to_log = None


def set_log_file(path):
    global file_to_log
    file_to_log = open(path, "a")


def printf(string=""):
    sys.stdout.write(string)
    sys.stdout.write("\n")
    sys.stdout.flush()

    if file_to_log:
        file_to_log.write(string)
        file_to_log.write("\n")
        file_to_log.flush()
