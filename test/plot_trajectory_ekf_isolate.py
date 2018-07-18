import numpy as np
import matplotlib.pyplot as plt
import config

sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

data_dir = config.save_path + "/ekf_debug/"

for i, sequence in enumerate(sequences):

    trajectories_to_overlay = [
        (data_dir + "%s_trajectory.npy" % sequence, {"linewidth": 1.0, "color": "r"}, "LiDAR Odometry"),
        (data_dir + "%s_ground_truth.npy" % sequence, {"linewidth": 1.0, "color": "b"}, "Ground Truth")
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

        y = trajectory[:, 1]
        x = trajectory[:, 0]

        plt.plot(x, y, **trj[1], label=trj[2])

    if not file_not_found:
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")

        if sequence in ["00", "01", "02", "08", "09"]:
            set_type = "Training Set"
        elif sequence in ["07"]:
            set_type = "Validation Set"
        else:
            set_type = "Test Set"

        plt.title("KITTI Sequence %s Trajectory (%s)" % (sequence, set_type))
        plt.legend()
        plt.savefig(data_dir + "%s_#00_fig_trajectory.png" % sequence)
        # plt.show()
        print("Plot saved for sequence %s" % sequence)

    # plot the ekf states
    file_not_found = False
    ekf_state_fp = data_dir + "%s_ekf_states.npy" % sequence
    fc_ground_truth_fp = data_dir + "%s_fc_ground_truth.npy" % sequence

    ekf_state_labels = ["Delta X", "Delta Y", "Delta Z",

                        "Velocity X", "Velocity Y", "Velocity Z",

                        "Pitch wrt Gravity", "Roll wrt Gravity",

                        "Accel Bias X", "Accel Bias Y", "Accel Bias Z",

                        "Delta Yaw", "Delta Pitch", "Delta Roll",

                        "Gyro Bias Yaw", "Gyro Bias Pitch", "Gyro Bias Roll",

                        "Delta X Ground Truth", "Delta Y Ground Truth",
                        "Delta Z Ground Truth",

                        "Delta Yaw Ground Truth", "Delta Pitch Ground Truth",
                        "Delta Roll Ground Truth"
                        ]
    ekf_state_units = ["m", "m", "m",
                       "m/s", "m/s", "m/s",
                       "rad", "rad",
                       "m/s^2", "m/s^2", "m/s^2",
                       "rad", "rad", "rad",
                       "rad/s", "rad/s", "rad/s",
                       "m", "m", "m",
                       "rad", "rad", "rad"]
    assert(len(ekf_state_labels) == len(ekf_state_units))
    try:
        ekf_states = np.load(ekf_state_fp)
        fc_ground_truth = np.load(fc_ground_truth_fp)
        ekf_states = np.concatenate([ekf_states, fc_ground_truth], axis=1)
    except FileNotFoundError:
        file_not_found = True
        print("Cannot find %s" % ekf_state_fp)
        continue

    if not file_not_found:
        for j in range(0, ekf_states.shape[1]):
            plt.figure(i)
            plt.clf()
            x = np.array(range(0, ekf_states.shape[0])) / 10.0
            y = ekf_states[:, j]

            plt.plot(x, y)

            plt.xlabel("time [s]")
            plt.ylabel("value [%s]" % ekf_state_units[j])
            plt.grid()

            plt.title("Seq. %s %s" % (sequence, ekf_state_labels[j]))
            plt.savefig(data_dir + "%s_#%02d_%s.png" % (sequence, j + 1, ekf_state_labels[j].lower().replace(" ", "_")))

        print("State plot saved for sequence %s" % sequence)
