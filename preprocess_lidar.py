# this script is to convert stored x y z reflectance lidar scans into images, and store them in a pickle for use when training

import pykitti
import numpy as np
from scipy.stats import binned_statistic
import pickle

base_dir = "/home/cs4li/Dev/KITTI/dataset/"
output_dir = "/home/cs4li/Dev/KITTI/dataset/sequences/lidar_pickles/"
# sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
sequences = ["04"]

# channel, height, width
img_shape = [64, 2000]
img_channels = [3]

enc_angles = np.linspace(-np.pi, np.pi, num=(img_shape[1] + 1), endpoint=False)

for seq in sequences:
    data = pykitti.odometry(base_dir, seq)
    length = [len(data.poses)]
    images = np.zeros(length + img_channels + img_shape, dtype=np.float32)
    scan_idx = 0

    # first need to convert each xyz
    for scan in data.velo:
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
                images[scan_idx, 0, 63 - i, :] = np.max(dist)
                images[scan_idx, 1, 63 - i, :] = 0
            else:
                strip_mean = binned_statistic(theta[az == i], [dist[az == i], scan[az == i, 3]], statistic='mean',
                                              bins=enc_angles)
                mask = np.isnan(strip_mean.statistic[0])

                # 0 is distance and 1 is intensity
                for j in range(0, 2):
                    # images[scan_idx, j, 63 - i, mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                    #                                               strip_mean.statistic[j, ~mask])
                    images[scan_idx, j, 63 - i, mask] = 0  # no interpolation, assign zero
                    images[scan_idx, j, 63 - i, ~mask] = strip_mean.statistic[j, ~mask]

                # save the mask
                images[scan_idx, 2, 63 - i, :] = ~mask

        if scan_idx % 100 == 0:
            print("Loading sequence %s %.1f%% " % (seq, (scan_idx / len(data.poses)) * 100))

        scan_idx = scan_idx + 1

    # save sequence to a pickle
    range_out = open(output_dir + str(seq) + "_range.pik", "wb")
    int_out = open(output_dir + str(seq) + "_intensity.pik", "wb")
    mask_out = open(output_dir + str(seq) + "_mask.pik", "wb")

    for i in range(0, len(data.poses)):
        pickle.dump(images[i, 0, :, :].astype(np.float16), range_out)
        pickle.dump((images[i, 1, :, :] * 255.0).astype(np.uint8), int_out)
        pickle.dump((images[i, 2, :, :]).astype(np.bool), mask_out)
        if i % 100 == 0:
            print("Saving sequence %s %.1f%% " % (seq, (i / len(data.poses)) * 100))

    range_out.close()
    int_out.close()
