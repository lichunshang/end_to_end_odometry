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


def printf(string=""):
    sys.stdout.write(string)
    sys.stdout.write("\n")
    sys.stdout.flush()


class Losses(object):
    def __init__(self, name, epochs, batches):
        self.name = name
        self.storage = np.zeros([epochs, batches], dtype=np.float32)

    def log(self, i_epoch, i_batch, val):
        self.storage[i_epoch, i_batch] = val

    def write_to_disk(self, path):
        np.save(os.path.join(path, self.name), self.storage)

    def get_val(self, i_epoch, i_batch):
        return self.storage[i_epoch, i_batch]

    def get_ave(self, i_epoch):
        return np.average(self.storage[i_epoch, :])


class LossesSet(object):
    def __init__(self, losses_names, epochs, batches):
        self.dict = collections.OrderedDict()
        for name in losses_names:
            self.dict[name] = Losses(name, epochs, batches)

    def log(self, losses_vals_map, i_epoch, i_batch):
        for key in self.dict:
            self.dict[key].log(i_epoch, i_batch, losses_vals_map[key])

    def batch_string(self, i_epoch, i_batch):
        string = ""
        for name in self.dict:
            val = self.dict[name].get_val(i_epoch, i_batch)
            string += "%s: %.3f, " % (name, val)

        return string

    def epoch_string(self, i_epoch):
        string = ""
        for name in self.dict:
            val = self.dict[name].get_ave(i_epoch)
            string += "ave_%s: %.3f, " % (name, val)

        return string

    def write_to_disk(self, path):
        for key in self.dict:
            self.dict[key].write_to_disk(path)
