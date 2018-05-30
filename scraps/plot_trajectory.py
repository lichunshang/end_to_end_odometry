import numpy as np
import matplotlib.pyplot as plt
import config

sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

data_dir = config.save_path + "/trajectory_results/"

for i, sequence in enumerate(sequences):

    trajectories_to_overlay = [
        (data_dir + "trajectory_%s.npy" % sequence, {"linewidth": 1.0, "color": "r"}, "LiDAR Odometry"),
        (data_dir + "ground_truth_%s.npy" % sequence, {"linewidth": 1.0, "color": "b"}, "Ground Truth")
    ]

    plt.figure(i)
    for trj in trajectories_to_overlay:
        trajectory = np.load(trj[0])

        z = trajectory[:, 2]
        x = trajectory[:, 0]

        plt.plot(x, z, **trj[1], label=trj[2])

    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("z [m]")

    if sequence in ["00", "01", "02", "08", "09"]:
        set_type = "Training Set"
    elif sequence in ["07"]:
        set_type = "Validation Set"
    else:
        set_type = "Test Set"

    plt.title("KITTI Sequence %s Trajectory (%s)" % (sequence, set_type))
    plt.legend()
    plt.savefig(data_dir + "fig_%s.png" % sequence)
    # plt.show()
