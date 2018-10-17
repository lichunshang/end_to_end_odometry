from scipy.misc import toimage
import os
import pickle
import config
import tools
import numpy as np

pickle_dir = config.kitti_lidar_pickles_path
# sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
sequences = ["00"]

for seq in sequences:
    cnt_intensity = 0
    cnt_range = 0
    cnt_mask = 0

    with (open(pickle_dir + seq + "_range.pik", "rb")) as opfile:
        while True:
            try:
                cur_image = pickle.load(opfile)
                curr_image = cur_image
                toimage(cur_image).save(tools.ensure_file_dir_exists(
                        os.path.join(pickle_dir, "%s_viz" % seq, "range", str(cnt_range) + ".png")))
                cnt_range += 1
            except EOFError:
                break

    with (open(pickle_dir + seq + "_intensity.pik", "rb")) as opfile:
        while True:
            try:
                cur_image = pickle.load(opfile)
                toimage(cur_image).save(tools.ensure_file_dir_exists(
                        os.path.join(pickle_dir, "%s_viz" % seq, "intensity", str(cnt_intensity) + ".png")))
                cnt_intensity += 1
            except EOFError:
                break

    with (open(pickle_dir + seq + "_mask.pik", "rb")) as opfile:
        while True:
            try:
                cur_image = pickle.load(opfile)
                cur_image = cur_image.astype(np.uint8)
                cur_image *= 255
                toimage(cur_image).save(tools.ensure_file_dir_exists(
                        os.path.join(pickle_dir, "%s_viz" % seq, "mask", str(cnt_mask) + ".png")))
                cnt_mask += 1
            except EOFError:
                break
