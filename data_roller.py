import pykitti
import numpy as np
import transformations
import tools
import random
import os
import pickle
import gc


class LidarDataLoader(object):
    def __init__(self, config, base_dir, seq, frames=None):
        pickles_dir = os.path.join(base_dir, "sequences", "lidar_pickles")
        seq_data = pykitti.odometry(base_dir, seq)
        num_frames = len(seq_data.poses)
        self.data = np.zeros([num_frames, 2, config.input_height, config.input_width], dtype=np.float16)

        with (open(os.path.join(pickles_dir, seq + "_range.pik"), "rb")) as opfile:
            i = 0
            while True:
                try:
                    cur_image = pickle.load(opfile)
                    self.data[i, 0, :, :] = cur_image
                except EOFError:
                    break
                i += 1
            assert (i == num_frames)

        with (open(os.path.join(pickles_dir, seq + "_intensity.pik"), "rb")) as opfile:
            i = 0
            while True:
                try:
                    cur_image = pickle.load(opfile)
                    self.data[i, 1, :, :] = cur_image
                    self.data[i, 1, :, :] = np.divide(self.data[i, 1, :, :], 254.0, dtype=np.float16)
                    self.data[i, 1, :, :] = np.subtract(self.data[i, 1, :, :], 0.5, dtype=np.float16)
                except EOFError:
                    break
                i += 1
            assert (i == num_frames)

        if frames:
            self.data = self.data[frames]

    def get(self, idx):
        return self.data[idx]


class StatefulRollerDataGen(object):

    # This version of the data generator slides along n frames at a time
    def __init__(self, config, base_dir, sequences, frames=None):
        self.cfg = config
        self.data_type = "cam"
        self.sequences = sequences
        self.curr_batch_idx = np.zeros([self.cfg.batch_size], dtype=np.int64)
        self.curr_batch_sequence = 0
        self.current_batch = 0
        self.sequence_batch = 0

        if (self.cfg.bidir_aug == True) and (self.cfg.batch_size % 2 != 0):
            raise ValueError("Batch size must be even")

        if hasattr(self.cfg, "data_type"):
            self.data_type = self.cfg.data_type
        if self.data_type != "cam" and self.data_type != "lidar":
            raise ValueError("lidar or camera for data type!")

        frames_data_type = np.uint8
        if self.data_type == "lidar":
            frames_data_type = np.float16

        if not frames:
            frames = [None] * len(sequences)

        self.input_frames = {}
        self.poses = {}

        # Mirror in y-z plane
        self.H = np.identity(3, dtype=np.float32)
        self.H[0][0] = -1.0

        self.se3_ground_truth = {}
        self.se3_mirror_ground_truth = {}
        self.fc_ground_truth = {}
        self.fc_reverse_ground_truth = {}
        self.fc_mirror_ground_truth = {}
        self.fc_reverse_mirror_ground_truth = {}
        self.batch_sizes = {}

        self.batch_cnt = 0
        self.total_frames = 0

        for i_seq, seq in enumerate(sequences):
            seq_data = pykitti.odometry(base_dir, seq, frames=frames[i_seq])
            lidar_data = None
            if self.data_type == "lidar":
                lidar_data = LidarDataLoader(self.cfg, base_dir, seq, frames=frames[i_seq])
            num_frames = len(seq_data.poses)

            self.input_frames[seq] = np.zeros(
                [num_frames, self.cfg.input_channels, self.cfg.input_height, self.cfg.input_width],
                dtype=frames_data_type)
            self.poses[seq] = seq_data.poses
            self.se3_ground_truth[seq] = np.zeros([num_frames, 7], dtype=np.float32)
            self.se3_mirror_ground_truth[seq] = np.zeros([num_frames, 7], dtype=np.float32)
            self.fc_ground_truth[seq] = np.zeros([num_frames - 1, 6], dtype=np.float32)
            self.fc_reverse_ground_truth[seq] = np.zeros([num_frames - 1, 6], dtype=np.float32)
            self.fc_mirror_ground_truth[seq] = np.zeros([num_frames - 1, 6], dtype=np.float32)
            self.fc_reverse_mirror_ground_truth[seq] = np.zeros([num_frames - 1, 6], dtype=np.float32)

            for i_img in range(num_frames):
                if i_img % 100 == 0:
                    tools.printf("Loading sequence %s %.1f%% " % (seq, (i_img / num_frames) * 100))

                # swap axis to channels first
                if self.data_type == "lidar":
                    img = lidar_data.get(i_img)
                else:
                    if self.cfg.input_channels == 1:
                        img = seq_data.get_cam0(i_img)
                    elif self.cfg.input_channels == 3:
                        img = seq_data.get_cam2(i_img)
                    else:
                        raise ValueError("Invalid number of channels for data")
                    img = img.resize((self.cfg.input_width, self.cfg.input_height))
                    img = np.array(img)
                    if self.cfg.input_channels == 3:
                        img = img[..., [2, 1, 0]]
                    img = np.reshape(img, [img.shape[0], img.shape[1], self.cfg.input_channels])
                    img = np.moveaxis(np.array(img), 2, 0)

                self.input_frames[seq][i_img, :] = img

                # now convert all the ground truth from 4x4 to xyz + quat, this is after the SE3 layer
                translation = transformations.translation_from_matrix(self.poses[seq][i_img])
                quat = transformations.quaternion_from_matrix(self.poses[seq][i_img])

                mirror_pose = np.identity(4, dtype=np.float32)
                mirror_pose[0:3, 0:3] = np.dot(self.H, np.dot(self.poses[seq][i_img][0:3, 0:3], self.H))
                mirror_pose[0:3, 3] = np.dot(self.H, translation)

                mirror_quat = transformations.quaternion_from_matrix(mirror_pose[0:3, 0:3])
                self.se3_ground_truth[seq][i_img] = np.concatenate([translation, quat])
                self.se3_mirror_ground_truth[seq][i_img] = np.concatenate([mirror_pose[0:3, 3], mirror_quat])

                # relative transformation labels
                if i_img + 1 < num_frames:
                    mirror_pose_next = np.identity(4, dtype=np.float32)
                    mirror_pose_next[0:3, 0:3] = np.dot(self.H, np.dot(self.poses[seq][i_img + 1][0:3, 0:3], self.H))
                    trans_next = transformations.translation_from_matrix(self.poses[seq][i_img + 1])
                    mirror_pose_next[0:3, 3] = np.dot(self.H, trans_next)

                    m_forward = np.dot(np.linalg.inv(self.poses[seq][i_img]), self.poses[seq][i_img + 1])
                    m_forward_mirror = np.dot(np.linalg.inv(mirror_pose), mirror_pose_next)
                    m_reverse = np.dot(np.linalg.inv(self.poses[seq][i_img + 1]), self.poses[seq][i_img])
                    m_reverse_mirror = np.dot(np.linalg.inv(mirror_pose_next), mirror_pose)

                    trans_forward = transformations.translation_from_matrix(m_forward)
                    ypr_forward = transformations.euler_from_matrix(m_forward, axes="rzyx")
                    trans_forward_mirror = transformations.translation_from_matrix(m_forward_mirror)
                    ypr_forward_mirror = transformations.euler_from_matrix(m_forward_mirror, axes="rzyx")
                    trans_reverse = transformations.translation_from_matrix(m_reverse)
                    ypr_reverse = transformations.euler_from_matrix(m_reverse, axes="rzyx")
                    trans_reverse_mirror = transformations.translation_from_matrix(m_reverse_mirror)
                    ypr_reverse_mirror = transformations.euler_from_matrix(m_reverse_mirror, axes="rzyx")

                    self.fc_ground_truth[seq][i_img] = np.concatenate([trans_forward, ypr_forward])
                    self.fc_mirror_ground_truth[seq][i_img] = np.concatenate([trans_forward_mirror, ypr_forward_mirror])
                    self.fc_reverse_ground_truth[seq][i_img] = np.concatenate([trans_reverse, ypr_reverse])
                    self.fc_reverse_mirror_ground_truth[seq][i_img] = np.concatenate(
                        [trans_reverse_mirror, ypr_reverse_mirror])

            # How many examples the sequence contains, were it to be processed using a batch size of 1
            sequence_examples = np.ceil((num_frames - self.cfg.timesteps) / self.cfg.sequence_stride).astype(
                np.int32)
            # how many batches in the sequence, cutting off any extra
            if self.cfg.bidir_aug:
                self.batch_sizes[seq] = np.floor(2 * sequence_examples / self.cfg.batch_size)
            else:
                self.batch_sizes[seq] = np.floor(sequence_examples / self.cfg.batch_size)

            self.batch_cnt += self.batch_sizes[seq].astype(np.int32)
            self.total_frames += num_frames

        tools.printf("All data loaded, batch_size=%d, timesteps=%d, num_batches=%d" % (
            self.cfg.batch_size, self.cfg.timesteps, self.batch_cnt))

        gc.collect()
        self.next_epoch()

    def next_batch(self):
        n = self.cfg.timesteps + 1  # number of frames in an example
        # prepare a batch from huge matrix of training data
        reset_state = [False, None]
        batch = np.zeros([n, self.cfg.batch_size, self.cfg.input_channels, self.cfg.input_height, self.cfg.input_width],
                         dtype=np.float32)
        se3_ground_truth = np.zeros([n, self.cfg.batch_size, 7], dtype=np.float32)
        fc_ground_truth = np.zeros([self.cfg.timesteps, self.cfg.batch_size, 6], dtype=np.float32)

        # check if at the end of the current sequence, and move to next if required
        if self.sequence_batch + 1 > self.batch_sizes[self.sequences[self.curr_batch_sequence]]:
            self.sequence_batch = 0
            self.curr_batch_sequence += 1
            self.set_batch_offsets()
            reset_state = [True, self.sequences[self.curr_batch_sequence]]

        for i_b in range(len(self.curr_batch_idx)):
            cur_seq = self.sequences[self.curr_batch_sequence]
            idx = self.curr_batch_idx[i_b]
            if (i_b < self.cfg.batch_size / 2) or not self.cfg.bidir_aug:
                # data going forwards
                batch[:, i_b, :, :, :] = self.input_frames[cur_seq][idx:idx + n, :, :, :]
                se3_ground_truth[:, i_b, :] = self.se3_ground_truth[cur_seq][idx:idx + n, :]
                fc_ground_truth[:, i_b, :] = self.fc_ground_truth[cur_seq][idx:idx + n - 1, :]
                self.curr_batch_idx[i_b] += self.cfg.sequence_stride
            else:
                # data going backwards
                if self.data_type == "lidar":
                    batch[:, i_b, :, :] = self.input_frames[cur_seq][idx - n:idx, :, :, :]
                    se3_ground_truth[:, i_b, :] = self.se3_mirror_ground_truth[cur_seq][idx - n:idx, :]
                    fc_ground_truth[:, i_b, :] = self.fc_reverse_mirror_ground_truth[cur_seq][idx - n:idx - 1, :]
                    # flip along time axis
                    batch[:, i_b, :, :] = np.flip(batch[:, i_b, :, :], axis=0)
                    # flip image along width axis
                    batch[:, i_b, :, :] = np.flip(batch[:, i_b, :, :], axis=3)
                    se3_ground_truth[:, i_b, :] = np.flip(se3_ground_truth[:, i_b, :], axis=0)
                    fc_ground_truth[:, i_b, :] = np.flip(fc_ground_truth[:, i_b, :], axis=0)
                else:
                    batch[:, i_b, :, :] = self.input_frames[cur_seq][idx - n:idx, :, :, :]
                    se3_ground_truth[:, i_b, :] = self.se3_ground_truth[cur_seq][idx - n:idx, :]
                    fc_ground_truth[:, i_b, :] = self.fc_reverse_ground_truth[cur_seq][idx - n:idx - 1, :]
                    # flip along time axis
                    batch[:, i_b, :, :] = np.flip(batch[:, i_b, :, :], axis=0)
                    se3_ground_truth[:, i_b, :] = np.flip(se3_ground_truth[:, i_b, :], axis=0)
                    fc_ground_truth[:, i_b, :] = np.flip(fc_ground_truth[:, i_b, :], axis=0)

                self.curr_batch_idx[i_b] -= self.cfg.sequence_stride

        if not self.data_type == "lidar":
            batch = np.divide(batch, 255.0, dtype=np.float32)  # ensure float32
            batch = np.subtract(batch, 0.5, dtype=np.float32)

        self.current_batch += 1
        self.sequence_batch += 1

        return reset_state, batch, fc_ground_truth, se3_ground_truth

    def has_next_batch(self):
        return self.current_batch < self.batch_cnt

    def set_batch_offsets(self):
        halfway = self.cfg.batch_size / 2
        halfway = int(halfway)
        for i_b in range(len(self.curr_batch_idx)):
            # first half of batches are going forward in time
            if (i_b < halfway) or not self.cfg.bidir_aug:
                self.curr_batch_idx[i_b] = self.batch_sizes[self.sequences[
                    self.curr_batch_sequence]] * i_b * self.cfg.sequence_stride
            # second half are going back in time
            else:
                reverse_start = self.batch_sizes[self.sequences[self.curr_batch_sequence]] * halfway * self.cfg.sequence_stride + 1
                self.curr_batch_idx[i_b] = reverse_start - self.batch_sizes[
                    self.sequences[self.curr_batch_sequence]] * (i_b - halfway) * self.cfg.sequence_stride

    def next_epoch(self):
        # Randomize sequence order
        random.shuffle(self.sequences)
        self.curr_batch_sequence = 0
        # set starting offsets for each batch
        self.set_batch_offsets()

        self.current_batch = 0
        self.sequence_batch = 0

    def curr_batch(self):
        return self.current_batch

    def total_batches(self):
        return self.batch_cnt

    def current_sequence(self):
        return self.sequences[self.curr_batch_sequence]

# lstm_states: dictionary with tuples of size 2, each element is [num_layers, batch_size, lstm_size]
# mask: tuple with bool whether to reset states or not, and the dictionary entry
def reset_select_lstm_state(lstm_states, mask, bidir_aug):
    if mask[0]:
        if bidir_aug:
            mid = lstm_states[mask[1]].shape[2] / 2
            mid = int(mid)
            lstm_states[mask[1]][:, :, 1:mid, :] = lstm_states[mask[1]][:, :, 0:(mid - 1), :]
            lstm_states[mask[1]][:, :, mid + 1:, :] = lstm_states[mask[1]][:, :, mid:-1, :]
            lstm_states[mask[1]][:, :, mid, :] = np.zeros(lstm_states[mask[1]][:, :, mid, :].shape,
                                                                dtype=np.float32)
        else:
            lstm_states[mask[1]][:, :, 1:, :] = lstm_states[mask[1]][:, :, 0:-1, :]
            lstm_states[mask[1]][:, :, 0, :] = np.zeros(lstm_states[mask[1]][:, :, 0, :].shape,
                                                          dtype=np.float32)
    return lstm_states


def reset_select_init_pose(init_pose, mask):
    for i in range(0, len(mask)):
        if mask[i]:
            init_pose[i, :] = np.array([0, 0, 0, 1, 0, 0, 0])
    return init_pose