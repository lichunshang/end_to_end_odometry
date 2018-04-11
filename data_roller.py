import pykitti
import numpy as np
import config
import gc
import transformations
import tools
import random


class StatefulRollerDataGen(object):

    # This version of the data generator slides along one frame at a time, starting neighbouring batches 1 step over
    # Sequence order is randomized
    def __init__(self, config, base_dir, sequences, frames=None):
        self.cfg = config
        self.sequences = sequences
        self.curr_batch_idx = np.arange(self.cfg.batch_size)
        self.curr_epoch_sequence = []
        self.current_batch = 0

        if not frames:
            frames = [None] * len(sequences)

        self.input_frames = {}
        self.poses = {}
        self.se3_ground_truth = {}
        self.fc_ground_truth = {}

        self.total_examples = 0

        for i_seq, seq in enumerate(sequences):
            seq_data = pykitti.odometry(base_dir, seq, frames=frames[i_seq])
            num_frames = len(seq_data.poses)

            self.input_frames[seq] = np.zeros([num_frames, self.cfg.input_channels, self.cfg.input_height, self.cfg.input_width])
            self.poses[seq] = seq_data.poses
            self.se3_ground_truth[seq] = np.zeros([num_frames, 7], dtype=np.float32)
            self.fc_ground_truth[seq] = np.zeros([num_frames - 1, 6], dtype=np.float32)

            for i_img in range(num_frames):
                if i_img % 100 == 0:
                    tools.printf("Loading sequence %s %.1f%% " % (seq, (i_img / num_frames) * 100))

                # swap axis to channels first
                img = seq_data.get_cam2(i_img)
                img = img.resize((self.cfg.input_width, self.cfg.input_height))
                img = np.array(img)
                if img.shape[2] == 3:  # convert to bgr if colored image
                    img = img[..., [2, 1, 0]]
                img = np.reshape(img, [img.shape[0], img.shape[1], self.cfg.input_channels])
                img = np.moveaxis(np.array(img), 2, 0)

                self.input_frames[seq][i_img, :] = img

                # now convert all the ground truth from 4x4 to xyz + quat, this is after the SE3 layer
                translation = transformations.translation_from_matrix(self.poses[seq][i_img])
                quat = transformations.quaternion_from_matrix(self.poses[seq][i_img])
                self.se3_ground_truth[seq][i_img] = np.concatenate([translation, quat])

                # relative transformation labels
                if (i_img + 1 < num_frames):
                    m = np.dot(np.linalg.inv(self.poses[seq][i_img]), self.poses[seq][i_img + 1])  # double check
                    translation = transformations.translation_from_matrix(m)
                    ypr = transformations.euler_from_matrix(m, axes="rzyx")
                    assert (np.all(np.abs(ypr) < np.pi))
                    self.fc_ground_truth[seq][i_img] = np.concatenate([translation, ypr])  # double check

            self.total_examples = self.total_examples + num_frames - self.cfg.timesteps + 1

        self.total_examples = self.total_examples - self.cfg.batch_size + 1

        tools.printf("All data loaded, batch_size=%d, timesteps=%d, num_batches=%d" % (
            self.cfg.batch_size, self.cfg.timesteps, self.total_examples))

        self.next_epoch()

    def next_batch(self):
        n = self.cfg.timesteps + 1  # number of frames in an example
        # prepare a batch from huge matrix of training data
        reset_state = np.zeros([self.cfg.batch_size], dtype=np.uint8)
        batch = np.zeros([n, self.cfg.batch_size, self.cfg.input_channels, self.cfg.input_height, self.cfg.input_width], dtype=np.float32)
        se3_ground_truth = np.zeros([n, self.cfg.batch_size, 7], dtype=np.float32)
        fc_ground_truth = np.zeros([n, self.cfg.batch_size, 6], dtype=np.float32)

        for i_b in range(len(self.curr_batch_idx)):
            # check if at the end of the current sequence, and move to next if required
            if self.curr_batch_idx[i_b] + self.cfg.timesteps > len(self.poses[self.curr_batch_sequences[i_b]]):
                self.curr_batch_idx[i_b] = 0
                self.curr_batch_sequences[i_b] = self.curr_batch_sequences[i_b] + 1
                reset_state[i_b] = 1
            batch[:,i_b,:,:,:] = self.input_frames[self.curr_epoch_sequence[self.curr_batch_sequences[i_b]]][self.curr_batch_idx[i_b]:self.curr_batch_idx[i_b] + n, :, :, :]
            self.curr_batch_idx[i_b] += 1

        batch = np.divide(batch, 255.0, dtype=np.float32)  # ensure float32

        return reset_state, batch, fc_ground_truth, se3_ground_truth

    def has_next_batch(self):
        return self.current_batch + 1 < self.total_examples

    def next_epoch(self):
        # set starting offsets for each batch
        self.curr_batch_idx = np.arange(self.cfg.batch_size, dtype=np.int)
        # Randomize sequence order
        self.curr_epoch_sequence = random.shuffle(self.sequences)
        self.curr_batch_sequences = np.zeros([self.cfg.batch_size])
        self.current_batch = 0

    def curr_batch(self):
        return self.current_batch

    def total_batches(self):
        return self.total_examples


# lstm_states: tuple of size 2, each element is [num_layers, batch_size, lstm_size]
# mask: vector of size batch_size, determines which example in the batch should have states reset
def reset_select_lstm_state(lstm_states, mask):
    # reset the lstm state for that example if indicated by the mask
    for i in range(0, len(mask)):
        if mask[i]:
            lstm_states[0][:, i, :] = 0
            lstm_states[1][:, i, :] = 0
    return lstm_states


def reset_select_init_pose(init_pose, mask):
    for i in range(0, len(mask)):
        if mask[i]:
            init_pose[i, :] = np.array([0, 0, 0, 1, 0, 0, 0])
    return init_pose
