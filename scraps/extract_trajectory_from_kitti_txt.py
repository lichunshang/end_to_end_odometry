import numpy as np

txt_path = "/home/cs4li/Downloads/15.txt"
dest_path = "/home/cs4li/Dev/end_to_end_visual_odometry/results/trajectory_results/ground_truth_15"

ground_truth = True

txt_file = open(txt_path)
poses = []

for line in txt_file:
    line = line.strip()
    line = line.split()

    x = float(line[0])
    y = float(line[1])

    pose = [x, y, 0, 1, 0, 0, 0]
    poses.append(pose)

poses = np.array(poses)
np.save(dest_path, poses)