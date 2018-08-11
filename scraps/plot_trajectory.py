import numpy as np
import matplotlib.pyplot as plt
import config

sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
save_dir = "/home/cs4li/Dev/end_to_end_odometry/results/train_seq_20180808-00-45-36_normalized_error_good_covar/"
# data_dir = config.save_path + "/trajectory_results/"
data_dir = save_dir + "/trajectory_results/"

for i, sequence in enumerate(sequences):

    trajectories_to_overlay = [
        (data_dir + "trajectory_%s.npy" % sequence, {"linewidth": 1.0, "color": "r"}, "LiDAR Odometry"),
        (data_dir + "ground_truth_%s.npy" % sequence, {"linewidth": 1.0, "color": "b"}, "Ground Truth")
    ]

    plt.figure(i)
    file_not_found = False
    for trj in trajectories_to_overlay:
        try:
            trajectory = np.load(trj[0])
        except FileNotFoundError:
            file_not_found = True
            print("Cannot find %s" % trj[0])
            break

        z = trajectory[:, 2]
        x = trajectory[:, 0]

        plt.plot(x, z, **trj[1], label=trj[2])

    if not file_not_found:
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
        print("Plot saved for sequence %s" % sequence)
