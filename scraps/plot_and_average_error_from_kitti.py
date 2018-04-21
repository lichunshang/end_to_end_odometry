import numpy as np
import matplotlib.pyplot as plt
import config
import os
import math

data_dir = os.path.join(config.save_path, "trajectory_results")

# gen average error plots
ave_xyz_error_wrt_path = np.loadtxt(os.path.join(data_dir, "kitti_evals", "plot_error", "avg_tl.txt"))
ave_rot_error_wrt_path = np.loadtxt(os.path.join(data_dir, "kitti_evals", "plot_error", "avg_rl.txt"))
ave_xyz_error_wrt_speed = np.loadtxt(os.path.join(data_dir, "kitti_evals", "plot_error", "avg_ts.txt"))
ave_rot_error_wrt_speed = np.loadtxt(os.path.join(data_dir, "kitti_evals", "plot_error", "avg_rs.txt"))

fig = plt.figure(1)
ax1 = fig.add_subplot(111)
ax1.plot(ave_xyz_error_wrt_path[:, 0], ave_xyz_error_wrt_path[:, 1] * 100, linewidth=1.0, color="r", marker="o",
         label="Translation Error")
ax2 = plt.twinx()
ax2.plot(ave_rot_error_wrt_path[:, 0], ave_rot_error_wrt_path[:, 1] * 180 / math.pi, linewidth=1.0, color="b",
         marker="s", label="Rotation Error")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2)
ax1.set_xlabel("Path Length [m]")
ax1.set_ylabel("Translation Error [%]")
ax2.set_ylabel("Rotation Error [deg/m]")
ax1.set_ylim(ymin=0)
ax2.set_ylim(ymin=0)
plt.title("Averaged Error vs. Path Length")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "fig_ave_err_vs_path_length.png"))

fig = plt.figure(2)
ax1 = fig.add_subplot(111)
ax1.plot(ave_xyz_error_wrt_speed[:, 0] * 3.6, ave_xyz_error_wrt_speed[:, 1] * 100, linewidth=1.0, color="r", marker="o",
         label="Translation Error")
ax2 = plt.twinx()
ax2.plot(ave_rot_error_wrt_speed[:, 0] * 3.6, ave_rot_error_wrt_speed[:, 1] * 180 / math.pi, linewidth=1.0, color="b",
         marker="s",
         label="Rotation Error")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2)
ax1.set_xlabel("Speed [km/h]")
ax1.set_ylabel("Translation Error [%]")
ax2.set_ylabel("Rotation Error [deg/m]")
ax1.set_ylim(ymin=0)
ax2.set_ylim(ymin=0)
plt.title("Averaged Error vs. Speed")
plt.tight_layout()
plt.savefig(os.path.join(data_dir, "fig_ave_err_vs_speed.png"))

# generate average translation and rotation error with respect to the path length for each seqeunce
sequences = ["03", "04", "05", "06", "07", "10"]
xyz_errors = []
rot_errors = []

for seq in sequences:
    xyz_error_wrt_path = np.loadtxt(os.path.join(data_dir, "kitti_evals", "plot_error", "%s_tl.txt" % seq))
    rot_error_wrt_path = np.loadtxt(os.path.join(data_dir, "kitti_evals", "plot_error", "%s_rl.txt" % seq))

    ave_xyz_error = np.average(xyz_error_wrt_path[:, 1])
    rot_xyz_error = np.average(rot_error_wrt_path[:, 1])

    xyz_errors.append(ave_xyz_error)
    rot_errors.append(rot_xyz_error)

    print("Seq %s ==> xyz: %5.2f [%%] rot: %6.4f [deg/m]" % (seq, ave_xyz_error * 100, rot_xyz_error * 180 / math.pi))

print("Averaged xyz: %5.2f [%%] rot: %6.4f [deg/m]" % (
    sum(xyz_errors) / len(xyz_errors) * 100, sum(rot_errors) / len(rot_errors) * 180 / math.pi))
# plt.show()
