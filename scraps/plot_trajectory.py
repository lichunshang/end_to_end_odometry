import numpy as np
import matplotlib.pyplot as plt

data_dir = "/home/cs4li/Dev/end_to_end_visual_odometry/results/trajectory_results/"

trajectories_to_overlay = [(data_dir + "trajectory_06.npy", {"linewidth": 1.0, "color": "r"},),
                           (data_dir + "ground_truth_06.npy", {"linewidth": 1.0, "color": "b"})]

plt.figure(1)
for trj in trajectories_to_overlay:
    trajectory = np.load(trj[0])

    x = trajectory[:, 0]
    y = trajectory[:, 1]

    plt.plot(x, y, **trj[1])

plt.show()
