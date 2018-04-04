import numpy as np
from scipy.misc import toimage

import pickle

pickle_dir = "/home/bapskiko/git/end_to_end_visual_odometry/pickles/"

sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

for seq in sequences:
    counter = 0
    with (open(pickle_dir + seq + "_intensity.pik", "rb")) as opfile:
        while True:
            try:
                cur_image = pickle.load(opfile)
                toimage(cur_image).save("/home/bapskiko/git/end_to_end_visual_odometry/intensity/" + seq + "_" + str(counter) + ".png")
                counter = counter + 1
            except EOFError:
                break

    with (open(pickle_dir + seq + "_range.pik", "rb")) as opfile:
        while True:
            try:
                cur_image = pickle.load(opfile)
                toimage(cur_image).save("/home/bapskiko/git/end_to_end_visual_odometry/range/" + seq + "_" + str(counter) + ".png")
                counter = counter + 1
            except EOFError:
                break



