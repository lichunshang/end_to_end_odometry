import pykitti
import numpy as np
import config
import gc
import transformations
import tools
import random


class StatefulDataGen(object):

    # Note some frames at the end of the sequence, and the
    # last sequence might be omitted to fit the examples
    # of timesteps x batch size8
    def __init__(self, config, base_dir, sequences, frames=None):
        self.truncated_seq_sizes = []
        self.end_of_sequence_indices = []
        self.curr_batch_idx = 0
        self.unused_batch_indices = []
        self.cfg = config

        if not frames:
            frames = [None] * len(sequences)

        total_num_examples = 0

        for i_seq, seq in enumerate(sequences):
            seq_data = pykitti.odometry(base_dir, seq, frames=frames[i_seq])
            num_frames = len(seq_data.poses)

            # less than timesteps number of frames will be discarded
            num_examples = (num_frames - 1) // self.cfg.timesteps
            self.truncated_seq_sizes.append(num_examples * self.cfg.timesteps + 1)
            total_num_examples += num_examples

        # less than batch size number of examples will be discarded
        self.total_batch_count = total_num_examples // self.cfg.batch_size
        # +1 adjusts for the extra image in the last time step
        total_timesteps = self.total_batch_count * (self.cfg.timesteps + 1)

        # since some examples will be discarded, readjust the truncated_seq_sizes
        deleted_frames = (total_num_examples - self.total_batch_count * self.cfg.batch_size) * self.cfg.timesteps
        for i in range(len(self.truncated_seq_sizes) - 1, -1, -1):
            if self.truncated_seq_sizes[i] > deleted_frames:
                self.truncated_seq_sizes[i] -= deleted_frames
                break
            else:
                self.truncated_seq_sizes[i] = 0
                deleted_frames -= self.truncated_seq_sizes[i]

        # for storing all training
        self.input_frames = np.zeros(
            [total_timesteps, self.cfg.batch_size, self.cfg.input_channels, self.cfg.input_height,
             self.cfg.input_width],
            dtype=np.uint8)
        poses_wrt_g = np.zeros([total_timesteps, self.cfg.batch_size, 4, 4], dtype=np.float32)  # ground truth poses

        num_image_loaded = 0
        for i_seq, seq in enumerate(sequences):
            seq_data = pykitti.odometry(base_dir, seq, frames=frames[i_seq])
            length = self.truncated_seq_sizes[i_seq]

            i = -1
            j = -1
            for i_img in range(length):

                if i_img % 100 == 0:
                    tools.printf("Loading sequence %s %.1f%% " % (seq, (i_img / length) * 100))

                i = num_image_loaded % total_timesteps
                j = num_image_loaded // total_timesteps

                # swap axis to channels first
                img = seq_data.get_cam0(i_img)
                img = img.resize((self.cfg.input_width, self.cfg.input_height))
                img = np.array(img)
                img = np.reshape(img, [img.shape[0], img.shape[1], self.cfg.input_channels])
                img = np.moveaxis(np.array(img), 2, 0)
                pose = seq_data.poses[i_img]

                self.input_frames[i, j] = img
                poses_wrt_g[i, j] = pose
                num_image_loaded += 1

                # insert the extra image at the end of the batch, note the number of
                # frames per batch of batch size 1 is timesteps + 1
                if i_img != 0 and i_img != length - 1 and i_img % self.cfg.timesteps == 0:
                    i = num_image_loaded % total_timesteps
                    j = num_image_loaded // total_timesteps
                    self.input_frames[i, j] = img
                    poses_wrt_g[i, j] = pose

                    num_image_loaded += 1

                gc.collect()  # force garbage collection

            # If the batch has the last frame in a sequence, the following frame
            # in the next batch must have a reset for lstm state
            self.end_of_sequence_indices.append((i + 1, j,))

        # make sure the all of examples are fully loaded, just to detect bugs
        assert (num_image_loaded == total_timesteps * self.cfg.batch_size)

        # now convert all the ground truth from 4x4 to xyz + quat, this is after the SE3 layer
        self.se3_ground_truth = np.zeros([total_timesteps, self.cfg.batch_size, 7], dtype=np.float32)
        for i in range(0, self.se3_ground_truth.shape[0]):
            for j in range(0, self.se3_ground_truth.shape[1]):
                translation = transformations.translation_from_matrix(poses_wrt_g[i, j])
                quat = transformations.quaternion_from_matrix(poses_wrt_g[i, j])
                self.se3_ground_truth[i, j] = np.concatenate([translation, quat])

        # extract the relative transformation between frames after the fully connected layer
        self.fc_ground_truth = np.zeros([total_timesteps, self.cfg.batch_size, 6], dtype=np.float32)
        # going through rows, then columns
        for i in range(0, self.fc_ground_truth.shape[0]):
            for j in range(0, self.fc_ground_truth.shape[1]):

                # always identity at the beginning of the sequence
                if i % (self.cfg.timesteps + 1) == 0:
                    m = transformations.identity_matrix()
                else:
                    m = np.dot(np.linalg.inv(poses_wrt_g[i - 1, j]), poses_wrt_g[i, j])  # double check

                translation = transformations.translation_from_matrix(m)
                ypr = transformations.euler_from_matrix(m, axes="rzyx")
                self.fc_ground_truth[i, j] = np.concatenate([translation, ypr])  # double check

        tools.printf("All data loaded, batch_size=%d, timesteps=%d, num_batches=%d" % (
            self.cfg.batch_size, self.cfg.timesteps, self.total_batch_count))

        self.next_epoch()

    def next_batch(self):
        i_b = self.curr_batch_idx
        n = self.cfg.timesteps + 1  # number of frames in an example
        # slice a batch from huge matrix of training data
        batch = self.input_frames[i_b * n: (i_b + 1) * n, :, :, :, :]
        batch = np.divide(batch, 255.0, dtype=np.float32)  # ensure float32

        se3_ground_truth = self.se3_ground_truth[i_b * n + 1: (i_b + 1) * n, :, :]
        fc_ground_truth = self.fc_ground_truth[i_b * n + 1: (i_b + 1) * n, :, :]
        init_poses = self.se3_ground_truth[i_b * n, :, :]

        # decide if we should propagate states
        i = self.curr_batch_idx * n
        reset_state = np.zeros([self.cfg.batch_size], dtype=np.uint8)
        for j in range(0, self.cfg.batch_size):
            if (i, j,) in self.end_of_sequence_indices:
                reset_state[j] = 1
            else:
                reset_state[j] = 0

        self.curr_batch_idx += 1

        return init_poses, reset_state, batch, fc_ground_truth, se3_ground_truth

    def next_batch_random(self):
        i_b = self.unused_batch_indices.pop()
        n = self.cfg.timesteps + 1  # number of frames in an example
        # slice a batch from huge matrix of training data
        batch = self.input_frames[i_b * n: (i_b + 1) * n, :, :, :, :]
        batch = np.divide(batch, 255.0, dtype=np.float32)  # ensure float32

        se3_ground_truth = self.se3_ground_truth[i_b * n + 1: (i_b + 1) * n, :, :]
        fc_ground_truth = self.fc_ground_truth[i_b * n + 1: (i_b + 1) * n, :, :]
        init_poses = self.se3_ground_truth[i_b * n, :, :]

        reset_state = np.ones([self.cfg.batch_size], dtype=np.uint8)
        self.curr_batch_idx += 1

        return init_poses, reset_state, batch, fc_ground_truth, se3_ground_truth

    def has_next_batch(self):
        return self.curr_batch_idx < self.total_batch_count

    def next_epoch(self):
        self.curr_batch_idx = 0

        # used for next_batch_random
        self.unused_batch_indices = range(0, self.total_batch_count)
        random.shuffle(self.unused_batch_indices)

    def curr_batch(self):
        return self.curr_batch_idx

    def total_batches(self):
        return self.total_batch_count


# lstm_states: tuple of size 2, each element is [num_layers, batch_size, lstm_size]
# mask: vector of size batch_size, determines which example in the batch should have states reset
def reset_select_lstm_state(lstm_states, mask):
    # reset the lstm state for that example if indicated by the mask
    for i in range(0, len(mask)):
        if mask[i]:
            lstm_states[0][:, i, :] = 0
            lstm_states[1][:, i, :] = 0
    return lstm_states
