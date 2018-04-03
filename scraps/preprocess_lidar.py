# this script is to convert stored x y z reflectance lidar scans into images, and store them in a pickle for use when training

import pykitti
import numpy as np
import pickle

base_dir = "/media/bapskiko/SpinDrive/kitti/dataset/"
sequences = ["00", "01", "02", "03"]

# channel, height, width
img_shape = [64, 1152]
img_channels = [2]

for seq in sequences:
    data = pykitti.odometry(base_dir, seq)

    length = [len(data.poses)]

    images = np.zeros(length + img_channels + img_shape, dtype=np.float32)

    #first need to convert each xyz
    for scan in data.velo:
        bin_stats = np.zeros()

        for pt in scan:
            print(pt)
