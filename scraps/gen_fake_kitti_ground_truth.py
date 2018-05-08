import os

poses_dir = "/home/cs4li/Dev/KITTI/dataset/poses"

sequences = ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
lengths = [921, 1061, 3281, 631, 1901, 1731, 491, 1801, 4981, 831, 2721]

identity_12 = "%f %f %f %f %f %f %f %f %f %f %f %f" % (1, 0, 0, 0,
                                                       0, 1, 0, 0,
                                                       0, 0, 1, 0)

for seq, length in zip(sequences, lengths):
    poses_file = open(os.path.join(poses_dir, "%s.txt" % seq), "w")

    poses_file.write("\n".join([identity_12] * length))

    poses_file.close()
