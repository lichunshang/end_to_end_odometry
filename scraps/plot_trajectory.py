import numpy as np
import matplotlib.pyplot as plt
import config

trajs_to_plt = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

for traj_i, traj in enumerate(trajs_to_plt):
    plt.figure(traj_i)
    trajectories_to_overlay = [
        (config.save_path + "trajectory_results/trajectory2_" + traj + ".npy", {"linewidth": 1.0, "color": "r"}, "timesteps 20 no reverse"),
        (config.save_path + "trajectory_results/trajectory10_" + traj + ".npy", {"linewidth": 1.0, "color": "g"},
         "timesteps 10 with reverse"),
        (config.save_path + "trajectory_results/ground_truth_" + traj + ".npy", {"linewidth": 1.0, "color": "b"}, "ground truth")
    ]
    for trj in trajectories_to_overlay:
        trajectory = np.load(trj[0])

        z = trajectory[:, 2]
        x = trajectory[:, 0]

        plt.plot(x, z, **trj[1], label=trj[2])
    plt.axis("equal")
    plt.legend()

plt.show()
