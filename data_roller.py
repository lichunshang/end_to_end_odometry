import pykitti
import numpy as np
import transformations
import tools
import random
import os
import pickle
import gc
import config
import scipy.constants


class LidarDataLoader(object):
    def __init__(self, cfg, base_dir, seq, frames=None):
        pickles_dir = config.lidar_pickles_path
        seq_data = pykitti.odometry(base_dir, seq)
        self.num_frames = len(seq_data.poses)
        self.data = np.zeros([self.num_frames, cfg.input_channels, cfg.input_height, cfg.input_width], dtype=np.float16)

        with (open(os.path.join(pickles_dir, seq + "_range.pik"), "rb")) as opfile:
            i = 0
            while True:
                try:
                    cur_image = pickle.load(opfile)
                    self.data[i, 0, :, :] = cur_image
                except EOFError:
                    break
                i += 1
                if i % 1000 == 0:
                    tools.printf("Loading lidar range seq. %s %.1f%% " % (seq, (i / self.num_frames) * 100))
            assert (i == self.num_frames)

        with (open(os.path.join(pickles_dir, seq + "_intensity.pik"), "rb")) as opfile:
            i = 0
            while True:
                try:
                    cur_image = pickle.load(opfile)
                    self.data[i, 1, :, :] = cur_image
                    self.data[i, 1, :, :] = np.divide(self.data[i, 1, :, :], 255.0, dtype=np.float16)
                    self.data[i, 1, :, :] = np.subtract(self.data[i, 1, :, :], 0.5, dtype=np.float16)
                except EOFError:
                    break
                i += 1
                if i % 1000 == 0:
                    tools.printf("Loading lidar intensity seq. %s %.1f%% " % (seq, (i / self.num_frames) * 100))
            assert (i == self.num_frames)

        # select the range of frames
        if frames:
            self.data = self.data[frames]
            self.num_frames = self.data.shape[0]

    def get(self, idx):
        return self.data[idx]


# to facilitate loading of raw data
class DataLoader(object):
    raw_range = {
        "00": (0, 4540),
        "01": (0, 1100),
        "02": (0, 4660),
        # "03": (0, 800),  # does not exist
        "04": (0, 270),
        "05": (0, 2760),
        "06": (0, 1100),
        "07": (0, 1100),
        "08": (1100, 5170),
        "09": (0, 1590),
        "10": (0, 1200),
    }

    raw_path = {
        "00": ("2011_10_03", "0027"),
        "01": ("2011_10_03", "0042"),
        "02": ("2011_10_03", "0034"),
        # "03": ("2011_09_26", "0067"),  # does not exist
        "04": ("2011_09_30", "0016"),
        "05": ("2011_09_30", "0018"),
        "06": ("2011_09_30", "0020"),
        "07": ("2011_09_30", "0027"),
        "08": ("2011_09_30", "0028"),
        "09": ("2011_09_30", "0033"),
        "10": ("2011_09_30", "0034"),
    }

    def __init__(self, cfg, base_dir, seq, frames=None):

        if seq == "03" or seq not in self.raw_path.keys():
            raise ValueError("%s sequence not supported" % seq)

        self.cfg = cfg
        self.imu_time_sync_shift = 0  # we discovered IMU seems to be delayed, so we shifted it forward

        if frames:
            range_start = self.raw_range[seq][0] + frames.start
            range_stop = range_start + (frames.stop - frames.start)
        else:
            range_start = self.raw_range[seq][0]
            range_stop = self.raw_range[seq][1] + 1

        self.data_raw_kitti = pykitti.raw(base_dir, self.raw_path[seq][0], self.raw_path[seq][1],
                                          frames=range(range_start, range_stop))
        self.data_imu_kitti = pykitti.raw(base_dir, self.raw_path[seq][0], self.raw_path[seq][1],
                                          frames=range(range_start + self.imu_time_sync_shift,
                                                       range_stop + self.imu_time_sync_shift))
        self.data_odom_kitti = pykitti.odometry(base_dir, seq, frames=frames)
        self.data_lidar_image = LidarDataLoader(self.cfg, base_dir, seq, frames=frames)

        assert (len(self.data_raw_kitti.oxts) == len(self.data_odom_kitti.poses) and
                len(self.data_imu_kitti.oxts) == len(self.data_odom_kitti.poses) and
                len(self.data_odom_kitti.poses) == self.data_lidar_image.num_frames)

        self.imu_measurements = []
        self.imu_measurements_reversed = []
        self.imu_timestamps = []

        self.__load_imu()

    def get_initial_states(self):
        return self.data_imu_kitti.oxts[0].packet

    def __load_imu(self):
        num_frames = self.get_num_frames()

        for i in range(0, num_frames):
            self.imu_timestamps.append(self.data_imu_kitti.timestamps[i])

            wx = self.data_imu_kitti.oxts[i].packet.wx
            wy = self.data_imu_kitti.oxts[i].packet.wy
            wz = self.data_imu_kitti.oxts[i].packet.wz
            ax = self.data_imu_kitti.oxts[i].packet.ax
            ay = self.data_imu_kitti.oxts[i].packet.ay
            az = self.data_imu_kitti.oxts[i].packet.az

            self.imu_measurements.append(np.array([wx, wy, wz, ax, ay, az, ]))

            # now we need to simulate IMU measurements in the reverse direction
            roll = self.data_imu_kitti.oxts[i].packet.roll
            pitch = self.data_imu_kitti.oxts[i].packet.pitch
            yaw = self.data_imu_kitti.oxts[i].packet.yaw

            # for accelerometer need to remove gravity vector and negate the values
            # first put the acceleration measurement in earth frame
            # Rotation of car imu (c) with respect to the ground truth (g)
            R_gc = transformations.euler_matrix(yaw, pitch, roll, axes="rzyx")[0:3, 0:3]
            R_gc_inv = np.linalg.inv(R_gc)

            g_accel_cc = R_gc.dot(np.array([ax, ay, az]))
            g_accel_cc[2] -= scipy.constants.g  # subtract gravity
            g_accel_cc = -g_accel_cc
            g_accel_cc[2] += scipy.constants.g  # add gravity back in
            c_accel_cc_reversed = R_gc_inv.dot(g_accel_cc)

            imu_reversed = np.array(
                    [-wx, -wy, -wz, c_accel_cc_reversed[0], c_accel_cc_reversed[1], c_accel_cc_reversed[2]])
            self.imu_measurements_reversed.append(imu_reversed)

    def get_cam_img(self, idx):
        if self.cfg.input_channels == 1:
            img = self.data_odom_kitti.get_cam0(idx)
        elif self.cfg.input_channels == 3:
            img = self.data_odom_kitti.get_cam2(idx)
        else:
            raise ValueError("Invalid number of channels for data")
        img = img.resize((self.cfg.input_width, self.cfg.input_height))
        img = np.array(img)
        if self.cfg.input_channels == 3:
            img = img[..., [2, 1, 0]]
        img = np.reshape(img, [img.shape[0], img.shape[1], self.cfg.input_channels])
        img = np.moveaxis(np.array(img), 2, 0)

        return img

    def get_lidar_img(self, idx):
        return self.data_lidar_image.get(idx)

    def get_img(self, idx):
        if self.cfg.data_type == "cam":
            return self.get_cam_img(idx)
        elif self.cfg.data_type == "lidar":
            return self.get_lidar_img(idx)

    def get_poses_in_corresponding_frame(self):
        if self.cfg.input_channels == 1:
            return self.data_odom_kitti.poses

        poses = self.data_odom_kitti.poses
        transformed_poses = []

        T_cam0_xxx = None
        if self.cfg.data_type == "cam" and self.cfg.input_channels == 3:
            T_cam2_cam0 = self.data_odom_kitti.calib.T_cam2_velo. \
                dot(np.linalg.inv(self.data_odom_kitti.calib.T_cam0_velo))
            T_cam0_xxx = np.linalg.inv(T_cam2_cam0)
        elif self.cfg.data_type == "lidar":
            T_cam0_xxx = self.data_odom_kitti.calib.T_cam0_velo

        for i in range(0, len(poses)):
            T_cam0_xxx_inv = np.linalg.inv(T_cam0_xxx)
            transformed_pose = np.dot(T_cam0_xxx_inv, np.dot(poses[i], T_cam0_xxx))
            transformed_poses.append(transformed_pose)

        return transformed_poses

    def get_num_frames(self):
        return len(self.data_odom_kitti.poses)

    def mirror_imu_on_xz(self, imu_meas):
        rate_imu = imu_meas[0:3]
        accel_imu = imu_meas[3:6]

        # reflection on xz
        H = np.eye(3, 3)
        H[1][1] = -1.0

        rate_imu = -H.dot(rate_imu)
        accel_imu = H.dot(accel_imu)

        return np.concatenate((rate_imu, accel_imu))

    def transform_imu_into_corresponding_frame(self, imu_meas):
        rate_c = imu_meas[0:3]
        accel_c = imu_meas[3:6]

        # transformation that maps measurement from car imu (c) to other sensor (x)
        T_xc = None
        if self.cfg.data_type == "cam" and self.cfg.input_channels == 1:
            T_xc = self.data_raw_kitti.calib.T_cam0_imu
        elif self.cfg.data_type == "cam" and self.cfg.input_channels == 3:
            T_xc = self.data_raw_kitti.calib.T_cam2_imu
        elif self.cfg.data_type == "lidar":
            T_xc = self.data_raw_kitti.calib.T_velo_imu

        # transforms angular rate to xxx
        rate_x = T_xc[0:3, 0:3].dot(rate_c)

        r = T_xc[0:3, 3]  # vector from imu to xxx
        w = rate_c
        a = accel_c
        # this transformation of linear acceleration assumes the angular acceleration is zero
        accel_x = T_xc[0:3, 0:3].dot(a + np.cross(w, np.cross(w, r)))

        return np.concatenate((rate_x, accel_x))

    def get_averaged_imu_in_corresponding_frame(self, idx, reverse=False, mirror=False):
        if reverse:
            imu_meas_t = self.imu_measurements_reversed[idx + 1]
            imu_meas_tm1 = self.imu_measurements_reversed[idx - 1]
        else:
            imu_meas_t = self.imu_measurements[idx]
            imu_meas_tm1 = self.imu_measurements[idx - 1]

        if mirror:
            imu_meas_t = self.mirror_imu_on_xz(imu_meas_t)
            imu_meas_tm1 = self.mirror_imu_on_xz(imu_meas_tm1)

        imu_meas_t = self.transform_imu_into_corresponding_frame(imu_meas_t)
        imu_meas_tm1 = self.transform_imu_into_corresponding_frame(imu_meas_tm1)

        # average the two imu measurements
        imu_meas_ave = (imu_meas_t + imu_meas_tm1) / 2

        return imu_meas_t


class StatefulRollerDataGen(object):

    # This version of the data generator slides along n frames at a time
    def __init__(self, cfg, base_dir, sequences, frames=None):
        self.cfg = cfg
        self.sequences = sequences
        self.curr_batch_sequence = 0
        self.current_batch = 0

        if (self.cfg.bidir_aug == True) and (self.cfg.batch_size % 2 != 0):
            raise ValueError("Batch size must be even")

        if self.cfg.data_type != "cam" and self.cfg.data_type != "lidar":
            raise ValueError("lidar or camera for data type!")

        frames_data_type = np.uint8
        if self.cfg.data_type == "lidar":
            frames_data_type = np.float16

        if not frames:
            frames = [None] * len(sequences)

        self.initial_states = {}
        self.input_frames = {}
        self.poses = {}

        # Mirror in y-z plane
        self.H = np.identity(3, dtype=np.float32)
        self.H[1][1] = -1.0

        self.se3_ground_truth = {}
        self.se3_mirror_ground_truth = {}
        self.fc_ground_truth = {}
        self.fc_reverse_ground_truth = {}
        self.fc_mirror_ground_truth = {}
        self.fc_reverse_mirror_ground_truth = {}

        self.imu_measurements = {}
        self.imu_measurements_mirror = {}
        self.imu_measurements_reverse = {}
        self.imu_measurements_reverse_mirror = {}
        self.imu_timestamps = {}

        # This keeps track of the number of batches in each sequence
        self.batch_counts = {}
        # this holds the batch ids for each sequence. It is randomized and the order determines the order in
        # which the batches are used
        self.batch_order = {}
        # this keeps track of the starting offsets in the input frames for batch id 0 in each sequence
        self.batch_offsets = {}
        # contains batch_cnt counters, each sequence is represented by it's index repeated for
        # however many batches are in that sequence
        self.sequence_ordering = None
        # contains batch_size counters. Keeps track of how many batches from each sequence
        # have already been used for the current epoch
        self.sequence_batch = {}

        self.batch_cnt = 0
        self.total_frames = 0

        for i_seq, seq in enumerate(sequences):
            seq_loader = DataLoader(self.cfg, base_dir, seq, frames=frames[i_seq])
            num_frames = seq_loader.get_num_frames()

            self.initial_states[seq] = seq_loader.get_initial_states()

            self.input_frames[seq] = np.zeros(
                    [num_frames, self.cfg.input_channels, self.cfg.input_height, self.cfg.input_width],
                    dtype=frames_data_type)
            self.poses[seq] = seq_loader.get_poses_in_corresponding_frame()
            self.se3_ground_truth[seq] = np.zeros([num_frames, 7], dtype=np.float32)
            self.se3_mirror_ground_truth[seq] = np.zeros([num_frames, 7], dtype=np.float32)
            self.fc_ground_truth[seq] = np.zeros([num_frames - 1, 6], dtype=np.float32)
            self.fc_reverse_ground_truth[seq] = np.zeros([num_frames - 1, 6], dtype=np.float32)
            self.fc_mirror_ground_truth[seq] = np.zeros([num_frames - 1, 6], dtype=np.float32)
            self.fc_reverse_mirror_ground_truth[seq] = np.zeros([num_frames - 1, 6], dtype=np.float32)

            self.imu_measurements[seq] = np.zeros([num_frames - 1, 6], dtype=np.float32)
            self.imu_measurements_mirror[seq] = np.zeros([num_frames - 1, 6], dtype=np.float32)
            self.imu_measurements_reverse[seq] = np.zeros([num_frames - 1, 6], dtype=np.float32)
            self.imu_measurements_reverse_mirror[seq] = np.zeros([num_frames - 1, 6], dtype=np.float32)
            self.imu_timestamps[seq] = np.zeros([num_frames], dtype=np.float64)

            for i_img in range(num_frames):
                if i_img % 100 == 0:
                    tools.printf("Loading sequence %s %.1f%% " % (seq, (i_img / num_frames) * 100))

                img = seq_loader.get_img(i_img)
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
                self.imu_timestamps[seq][i_img] = seq_loader.imu_timestamps[i_img].timestamp()

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

                    get_imu = seq_loader.get_averaged_imu_in_corresponding_frame
                    self.imu_measurements[seq][i_img] = get_imu(i_img, mirror=False, reverse=False)
                    self.imu_measurements_mirror[seq][i_img] = get_imu(i_img, mirror=True, reverse=False)
                    self.imu_measurements_reverse[seq][i_img] = get_imu(i_img, mirror=False, reverse=True)
                    self.imu_measurements_reverse_mirror[seq][i_img] = get_imu(i_img, mirror=True, reverse=True)

            self.imu_timestamps[seq] = np.ediff1d(self.imu_timestamps[seq])

            # How many examples the sequence contains, were it to be processed using a batch size of 1
            sequence_examples = np.ceil((num_frames - self.cfg.timesteps) / self.cfg.sequence_stride).astype(
                    np.int32)
            # how many batches in the sequence, cutting off any extra
            if self.cfg.bidir_aug:
                self.batch_counts[seq] = np.floor(2 * sequence_examples / self.cfg.batch_size).astype(np.int32)
            else:
                self.batch_counts[seq] = np.floor(sequence_examples / self.cfg.batch_size).astype(np.int32)

            self.batch_cnt += self.batch_counts[seq].astype(np.int32)
            self.batch_order[seq] = np.arange(0, self.batch_counts[seq], dtype=np.uint32)
            self.batch_offsets[seq] = np.zeros([self.cfg.batch_size], dtype=np.uint32)
            self.sequence_batch[seq] = 0

            self.total_frames += num_frames

        tools.printf("All data loaded, batch_size=%d, timesteps=%d, num_batches=%d" % (
            self.cfg.batch_size, self.cfg.timesteps, self.batch_cnt))

        gc.collect()
        self.sequence_ordering = np.zeros([self.batch_cnt], dtype=np.uint16)
        self.set_batch_offsets()
        self.next_epoch(False)

    # get the first pose of a given sequence
    def get_sequence_initial_pose(self, seq):
        return self.se3_ground_truth[seq][0, :]

    def next_batch(self):
        n = self.cfg.timesteps + 1  # number of frames in an example
        # prepare a batch from huge matrix of training data
        batch = np.zeros([n, self.cfg.batch_size, self.cfg.input_channels, self.cfg.input_height, self.cfg.input_width],
                         dtype=np.float32)
        se3_ground_truth = np.zeros([n, self.cfg.batch_size, 7], dtype=np.float32)
        fc_ground_truth = np.zeros([self.cfg.timesteps, self.cfg.batch_size, 6], dtype=np.float32)
        imu_measurements = np.zeros([self.cfg.timesteps, self.cfg.batch_size, 6], dtype=np.float32)
        imu_dt = np.zeros([n - 1, self.cfg.batch_size], dtype=np.float64)

        # check if at the end of the current sequence, and move to next if required
        cur_seq = self.sequences[self.sequence_ordering[self.current_batch]]
        # lookup batch id give how many batches in the sequence have already been processed
        batch_id = self.batch_order[cur_seq][self.sequence_batch[cur_seq]]

        start_idx = self.batch_offsets[cur_seq]
        idx_offset = self.cfg.sequence_stride * batch_id

        for i_b in range(self.cfg.batch_size):
            if (i_b < self.cfg.batch_size / 2) or not self.cfg.bidir_aug:
                # data going forwards
                idx = start_idx[i_b] + idx_offset
                batch[:, i_b, :, :, :] = self.input_frames[cur_seq][idx:idx + n, :, :, :]
                se3_ground_truth[:, i_b, :] = self.se3_ground_truth[cur_seq][idx:idx + n, :]
                fc_ground_truth[:, i_b, :] = self.fc_ground_truth[cur_seq][idx:idx + n - 1, :]
                imu_measurements[:, i_b, :] = self.imu_measurements[cur_seq][idx:idx + n - 1, :]
                imu_dt[:, i_b] = self.imu_timestamps[cur_seq][idx:idx + n - 1]
            else:
                # data going backwards
                idx = start_idx[i_b] - idx_offset
                if self.cfg.data_type == "lidar":
                    batch[:, i_b, :, :] = self.input_frames[cur_seq][idx - n:idx, :, :, :]
                    se3_ground_truth[:, i_b, :] = self.se3_mirror_ground_truth[cur_seq][idx - n:idx, :]
                    fc_ground_truth[:, i_b, :] = self.fc_reverse_mirror_ground_truth[cur_seq][idx - n:idx - 1, :]
                    imu_measurements[:, i_b, :] = self.imu_measurements_reverse_mirror[cur_seq][idx - n:idx - 1, :]
                    # flip along time axis
                    batch[:, i_b, :, :] = np.flip(batch[:, i_b, :, :], axis=0)
                    # flip image along width axis
                    batch[:, i_b, :, :] = np.flip(batch[:, i_b, :, :], axis=3)
                    se3_ground_truth[:, i_b, :] = np.flip(se3_ground_truth[:, i_b, :], axis=0)
                    fc_ground_truth[:, i_b, :] = np.flip(fc_ground_truth[:, i_b, :], axis=0)
                    imu_measurements[:, i_b, :] = np.flip(imu_measurements[:, i_b, :], axis=0)
                    imu_dt[:, i_b] = np.flip(self.imu_timestamps[cur_seq][idx - n:idx - 1], axis=0)
                else:
                    batch[:, i_b, :, :] = self.input_frames[cur_seq][idx - n:idx, :, :, :]
                    se3_ground_truth[:, i_b, :] = self.se3_ground_truth[cur_seq][idx - n:idx, :]
                    fc_ground_truth[:, i_b, :] = self.fc_reverse_ground_truth[cur_seq][idx - n:idx - 1, :]
                    imu_measurements[:, i_b, :] = self.imu_measurements_reverse[cur_seq][idx - n:idx - 1, :]
                    # flip along time axis
                    batch[:, i_b, :, :] = np.flip(batch[:, i_b, :, :], axis=0)
                    se3_ground_truth[:, i_b, :] = np.flip(se3_ground_truth[:, i_b, :], axis=0)
                    fc_ground_truth[:, i_b, :] = np.flip(fc_ground_truth[:, i_b, :], axis=0)
                    imu_measurements[:, i_b, :] = np.flip(imu_measurements[:, i_b, :], axis=0)
                    imu_dt[:, i_b] = np.flip(self.imu_timestamps[cur_seq][idx - n:idx - 1], axis=0)

        if self.cfg.data_type == "cam":
            batch = np.divide(batch, 255.0, dtype=np.float32)  # ensure float32
            batch = np.subtract(batch, 0.5, dtype=np.float32)

        self.current_batch += 1
        self.sequence_batch[cur_seq] += 1

        return batch_id, cur_seq, batch, fc_ground_truth, se3_ground_truth, imu_measurements, imu_dt.astype(np.float32)

    def has_next_batch(self):
        return self.current_batch < self.batch_cnt

    def set_batch_offsets(self):
        halfway = self.cfg.batch_size / 2
        halfway = int(halfway)
        start = 0
        for i_seq, seq in enumerate(self.sequences):
            self.sequence_ordering[start:(self.batch_counts[seq] + start)] = i_seq
            start += self.batch_counts[seq]

            reverse_start = self.batch_counts[seq] * halfway * self.cfg.sequence_stride + 1
            for i_b in range(self.cfg.batch_size):
                # first half of batches are going forward in time
                if (i_b < halfway) or not self.cfg.bidir_aug:
                    start_idx = self.batch_counts[seq] * i_b * self.cfg.sequence_stride
                    self.batch_offsets[seq][i_b] = start_idx
                # second half are going back in time
                else:
                    end_idx = reverse_start - self.batch_counts[seq] * (i_b - halfway) * self.cfg.sequence_stride
                    self.batch_offsets[seq][i_b] = end_idx

    def next_epoch(self, randomize=True):
        # Randomize sequence order
        if randomize:
            random.shuffle(self.sequence_ordering)
            # randomize batch order in each sequence
            for seq in self.sequences:
                np.random.shuffle(self.batch_order[seq])

        self.current_batch = 0
        for seq in self.sequences:
            self.sequence_batch[seq] = 0

    def curr_batch(self):
        return self.current_batch

    def total_batches(self):
        return self.batch_cnt

    def current_sequence(self):
        return self.sequences[self.curr_batch_sequence]

    def get_initial_state(self, seq):
        return self.initial_states[seq]


# lstm_states: dictionary indexed by sequence ids, each elem being [batch_count, 2, num_layers, batch_size, lstm_size]
# lstm_init_states: a [2, num_layers, batch_size, lstm_size] object containing initialization for the selected batches
# returns selected_lstm_states
# state_idx: list of states to update each element is [sequence, example_index]

def update_lstm_state(lstm_states, lstm_update, cur_seq, batch_id):
    lstm_states[cur_seq][batch_id, ...] = lstm_update


def get_init_lstm_state(lstm_states, lstm_init_states, cur_seq, batch_id, bidir_aug):
    if batch_id > 0:
        lstm_init_states = lstm_states[cur_seq][batch_id - 1, ...]
    else:
        if bidir_aug:
            mid = lstm_init_states.shape[2] / 2
            mid = int(mid)
            lstm_init_states[:, :, 0, :] = np.zeros(lstm_init_states[:, :, 0, :].shape, dtype=np.float32)
            lstm_init_states[:, :, 1:mid, :] = lstm_states[cur_seq][-1, :, :, 0:(mid - 1), :]
            lstm_init_states[:, :, mid, :] = np.zeros(lstm_init_states[:, :, 0, :].shape, dtype=np.float32)
            lstm_init_states[:, :, mid + 1:, :] = lstm_states[cur_seq][-1, :, :, mid:-1, :]
        else:
            lstm_init_states[:, :, 0, :] = np.zeros(lstm_init_states[:, :, 0, :].shape, dtype=np.float32)
            lstm_init_states[:, :, 1:, :] = lstm_states[cur_seq][-1, :, :, 0:-1, :]


def update_ekf_state(ekf_states, ekf_cov_states, ekf_update, ekf_cov_update, cur_seq, batch_id):
    ekf_states[cur_seq][batch_id, ...] = ekf_update
    ekf_cov_states[cur_seq][batch_id, ...] = ekf_cov_update


def get_init_ekf_states(ekf_states, ekf_cov_states, ekf_init_states, ekf_cov_init_states, cur_seq, batch_id, bidir_aug):
    if batch_id > 0:
        ekf_init_states = ekf_states[cur_seq][batch_id - 1, ...]
        ekf_cov_init_states = ekf_cov_states[cur_seq][batch_id - 1, ...]
    else:
        if bidir_aug:
            mid = ekf_init_states.shape[0] / 2
            mid = int(mid)
            ekf_init_states[0, ...] = np.zeros(ekf_init_states[0, ...].shape, dtype=np.float32)
            ekf_init_states[1:mid, ...] = ekf_states[cur_seq][-1, 0:(mid - 1), ...]
            ekf_init_states[mid, ...] = np.zeros(ekf_init_states[0, ...].shape, dtype=np.float32)
            ekf_init_states[mid + 1:, ...] = ekf_states[cur_seq][-1, mid:-1, ...]

            ekf_cov_init_states[0, ...] = np.identity(ekf_init_states[0, ...].shape[0], dtype=np.float32)
            ekf_cov_init_states[1:mid, ...] = ekf_cov_states[cur_seq][-1, 0:(mid - 1), ...]
            ekf_cov_init_states[mid, ...] = np.identity(ekf_init_states[0, ...].shape[0], dtype=np.float32)
            ekf_cov_init_states[mid + 1:, ...] = ekf_cov_states[cur_seq][-1, mid:-1, ...]
        else:
            ekf_init_states[0, ...] = np.zeros(ekf_init_states[0, ...].shape, dtype=np.float32)
            ekf_init_states[1:, ...] = ekf_states[cur_seq][-1, 0:-1, ...]

            ekf_cov_init_states[0, ...] = np.identity(ekf_init_states[0, ...].shape[0], dtype=np.float32)
            ekf_cov_init_states[1:, ...] = ekf_cov_states[cur_seq][-1, 0:-1, ...]


def reset_select_init_pose(init_pose, mask):
    for i in range(0, len(mask)):
        if mask[i]:
            init_pose[i, :] = np.array([0, 0, 0, 1, 0, 0, 0])
    return init_pose
