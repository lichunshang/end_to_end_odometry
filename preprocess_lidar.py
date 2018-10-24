# this script is to convert stored x y z reflectance lidar scans into images, and store them in a pickle for use when training

import pykitti
import numpy as np
from scipy.stats import binned_statistic
import pickle
import config
import tools
import cv2
import os
import data_roller

data_source = "KITTI_raw"

output_dir = config.kitti_lidar_pickles_path
sequences = ["00", "01", "02", "04", "05", "06", "07", "08", "09", "10"]
# sequences = ["00"]

# modes
# one of horizontal_interp, no_interp, depth_completion
HORIZONTAL_INTERP = 0
NO_INTERP = 1
DEPTH_COMPLETION = 2
mode = NO_INTERP

# channel, height, width
img_shape = [64, 1152]
img_channels = [4]

enc_angles = np.linspace(-np.pi, np.pi, num=(img_shape[1] + 1), endpoint=False)
averaged_masks = []

for seq in sequences:

    if data_source == "KITTI":
        base_dir = config.dataset_path
        data_kitti_odom = pykitti.odometry(base_dir, seq, frames=None)
        length = len(data_kitti_odom.poses)
        get_scan = lambda idx: data_kitti_odom.get_velo(idx)
        get_timestamp = lambda idx: data_kitti_odom.timestamps[idx]
    elif data_source == "KITTI_raw":
        base_dir = config.dataset_path
        data_kitti_raw = data_roller.map_kitti_raw_to_odometry(base_dir, seq, frames=None)
        length = len(data_kitti_raw.timestamps)
        get_scan = lambda idx: data_kitti_raw.get_velo(idx)
        get_timestamp = lambda idx: data_kitti_raw.timestamps[idx]
    else:
        raise NotImplementedError("%s data source not valid" % data_source)

    data = pykitti.odometry(base_dir, seq, frames=None)

    images = np.zeros([length] + img_channels + img_shape, dtype=np.float32)
    scan_idx = 0

    # first need to convert each xyz
    for i in range(0, length):
        scan = get_scan(i)
        theta = np.arctan2(scan[:, 1], scan[:, 0])
        xy = np.sqrt(np.square(scan[:, 0]) + np.square(scan[:, 1]))
        az = np.arctan2(scan[:, 2], xy)

        velo_start = np.min(az)
        velo_end = np.max(az)
        spacing = (velo_end - velo_start) / 63
        az = np.rint((az - velo_start) / spacing).astype(np.int16)

        dist = np.sqrt(np.square(xy) + np.square(scan[:, 2]))

        for i in range(0, 64):
            if len(theta[az == i]) == 0:
                if mode == HORIZONTAL_INTERP:
                    images[scan_idx, 0, 63 - i, :] = np.max(dist)
                else:
                    images[scan_idx, 0, 63 - i, :] = 0
                images[scan_idx, 1, 63 - i, :] = 0
            else:
                strip_mean = binned_statistic(theta[az == i], [dist[az == i], scan[az == i, 3]], statistic='mean',
                                              bins=enc_angles)
                mask = np.isnan(strip_mean.statistic[0])

                # 0 is distance and 1 is intensity
                for j in range(0, 2):
                    if mode == HORIZONTAL_INTERP:
                        images[scan_idx, j, 63 - i, mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                                                                      strip_mean.statistic[j, ~mask])
                    else:
                        images[scan_idx, j, 63 - i, mask] = 0
                    images[scan_idx, j, 63 - i, ~mask] = strip_mean.statistic[j, ~mask]

                # save the mask
                images[scan_idx, 2, 63 - i, :] = ~mask

            # timestamp image
            if data_source == "KITTI":
                images[scan_idx, 3, :, :] = get_timestamp(i)
            elif data_source == "KITTI_raw":
                images[scan_idx, 3, :, :] = i * 0.1 + np.linspace(0, 0.1, num=img_shape[1], endpoint=False)
            else:
                raise NotImplementedError("%s data source not valid" % data_source)

        if scan_idx % 100 == 0:
            print("Loading sequence %s %.1f%% " % (seq, (scan_idx / len(data.poses)) * 100))

        scan_idx = scan_idx + 1

    # find out the parts of the lidar where is almost always has no return
    # get the percentage of no return for each pixel
    averaged_mask = np.average(images[:, 2, :, :], axis=0)
    averaged_masks.append(averaged_mask)

    # save sequence to a pickle
    range_out = open(tools.ensure_file_dir_exists(os.path.join(output_dir, str(seq) + "_range.pik")), "wb")
    int_out = open(tools.ensure_file_dir_exists(os.path.join(output_dir, str(seq) + "_intensity.pik")), "wb")
    mask_out = open(tools.ensure_file_dir_exists(os.path.join(output_dir, str(seq) + "_mask.pik")), "wb")
    time_out = open(tools.ensure_file_dir_exists(os.path.join(output_dir, str(seq) + "_time.pik")), "wb")

    for i in range(0, len(data.poses)):
        # need to flip them horizontally because the bins are from -pi to pi, we want the
        # image to be from pi (left most) to -pi (right most)
        pickle.dump(np.fliplr(images[i, 0, :, :].astype(np.float16)), range_out)
        pickle.dump(np.fliplr((images[i, 1, :, :] * 255.0).astype(np.uint8)), int_out)
        pickle.dump(np.fliplr((images[i, 2, :, :]).astype(np.bool)), mask_out)
        pickle.dump(images[i, 3, :, :], time_out)
        if i % 100 == 0:
            print("Saving sequence %s %.1f%% " % (seq, (i / len(data.poses)) * 100))

    range_out.close()
    int_out.close()
    mask_out.close()
    time_out.close()

if mode == DEPTH_COMPLETION:
    # now we re-read the the saved pickles again to apply the mask and depth completion
    no_ret_mask = np.fliplr(np.average(np.stack(averaged_masks), axis=0) < 0.1)
    for seq in sequences:
        images = []
        with (open(os.path.join(output_dir, str(seq) + "_range.pik"), "rb")) as opfile:
            while True:
                try:
                    images.append(pickle.load(opfile))
                except EOFError:
                    break
        images = np.stack(images, axis=0)

        range_out = open(tools.ensure_file_dir_exists(os.path.join(output_dir, str(seq) + "_range.pik")), "wb")
        for i in range(0, images.shape[0]):
            depth_map = images[i, :, :].astype(np.float32)
            depth_map[no_ret_mask] = 0
            max_depth = 100.0

            # do depth completion
            # invert
            valid_pixels = (depth_map > 0.1)
            depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]

            empty_pixels = depth_map < 0.1
            op = cv2.dilate(depth_map, np.ones((1, 3)))
            op = cv2.morphologyEx(op, cv2.MORPH_CLOSE, np.ones((2, 2)))
            op = cv2.dilate(depth_map, np.ones((2, 2)))

            op_empty = op < 0.1
            op_dilated = cv2.dilate(op, np.ones((3, 7)))
            op[op_empty] = op_dilated[op_empty]

            op_empty = op < 0.1
            op_dilated = cv2.dilate(op, np.ones((5, 9)))
            op[op_empty] = op_dilated[op_empty]

            op_empty = op < 0.1
            op_dilated = cv2.dilate(op, np.ones((9, 15)))
            op[op_empty] = op_dilated[op_empty]

            op_empty = op < 0.1
            op_dilated = cv2.dilate(op, np.ones((31, 31)))
            op[op_empty] = op_dilated[op_empty]

            op = cv2.medianBlur(op, 3)
            op = cv2.blur(op, (3, 3), borderType=cv2.BORDER_REPLICATE)

            depth_map[empty_pixels] = op[empty_pixels]

            # invert back
            valid_pixels = (depth_map > 0.1)
            depth_map[valid_pixels] = max_depth - depth_map[valid_pixels]
            depth_map[no_ret_mask] = 0
            images[i, :, :] = depth_map.astype(np.float16)

            pickle.dump(images[i, :, :].astype(np.float16), range_out)
            if i % 100 == 0: print("Saving sequence %s %.1f%% " % (seq, (i / len(data.poses)) * 100))
