import numpy as np
import matplotlib.pyplot as plt
import transformations
import config
import os
import sys

sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
save_dir = sys.argv[1]
# data_dir = config.save_path + "/trajectory_results/"
data_dir = os.path.join(save_dir, "trajectory_results")

for i, sequence in enumerate(sequences):

    file_not_found = False
    try:
        trajectory = np.load(os.path.join(data_dir, "%s_trajectory.npy" % sequence))
        trajectory_gt = np.load(os.path.join(data_dir, "%s_ground_truth.npy" % sequence))
    except FileNotFoundError:
        file_not_found = True
        print("Cannot find trajectory for seq %s" % sequence)
        continue

    x = trajectory[:, 0]
    y = trajectory[:, 1]
    z = trajectory[:, 2]
    euler = np.stack(
            [transformations.euler_from_quaternion(trajectory[j, 3:], axes="rzyx") for j in range(trajectory.shape[0])])
    yaw = euler[:, 0]
    pitch = euler[:, 1]
    roll = euler[:, 2]

    x_gt = trajectory_gt[:, 0]
    y_gt = trajectory_gt[:, 1]
    z_gt = trajectory_gt[:, 2]
    euler_gt = np.stack([transformations.euler_from_quaternion(trajectory_gt[j, 3:], axes="rzyx") for j in
                         range(trajectory.shape[0])])
    yaw_gt = euler_gt[:, 0]
    pitch_gt = euler_gt[:, 1]
    roll_gt = euler_gt[:, 2]

    plt.figure(1)
    plt.clf()
    plt.plot(x, y, linewidth=1.0, color="r", label="Estimate")
    plt.plot(x_gt, y_gt, linewidth=1.0, color="b", label="Ground Truth")
    plt.figure(2)
    plt.clf()
    plt.plot(x, z, linewidth=1.0, color="r", label="Estimate")
    plt.plot(x_gt, z_gt, linewidth=1.0, color="b", label="Ground Truth")
    plt.figure(3)
    plt.clf()
    plt.plot(y, z, linewidth=1.0, color="r", label="Estimate")
    plt.plot(y_gt, z_gt, linewidth=1.0, color="b", label="Ground Truth")
    plt.figure(4)
    plt.clf()
    plt.plot(np.unwrap(yaw), linewidth=1.0, color="r", label="Estimate")
    plt.plot(np.unwrap(yaw_gt), linewidth=1.0, color="b", label="Ground Truth")
    plt.figure(5)
    plt.clf()
    plt.plot(np.unwrap(pitch), linewidth=1.0, color="r", label="Estimate")
    plt.plot(np.unwrap(pitch_gt), linewidth=1.0, color="b", label="Ground Truth")
    plt.figure(6)
    plt.clf()
    plt.plot(np.unwrap(roll), linewidth=1.0, color="r", label="Estimate")
    plt.plot(np.unwrap(roll_gt), linewidth=1.0, color="b", label="Ground Truth")

    if not file_not_found:
        plt.figure(1)
        plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title("KITTI Seq. %s Trajectory XY Plane" % sequence)
        plt.legend()
        plt.savefig(data_dir + "%s_#00_0_fig_xy_trajectory.png" % sequence)

        plt.figure(2)
        # plt.axis("equal")
        plt.xlabel("x [m]")
        plt.ylabel("z [m]")
        plt.title("KITTI Seq. %s Trajectory XZ Plane" % sequence)
        plt.legend()
        plt.savefig(data_dir + "%s_#00_1_fig_xz_trajectory.png" % sequence)

        plt.figure(3)
        # plt.axis("equal")
        plt.xlabel("y [m]")
        plt.ylabel("z [m]")
        plt.title("KITTI Seq. %s Trajectory YZ Plane" % sequence)
        plt.legend()
        plt.savefig(data_dir + "%s_#00_2_fig_yz_trajectory.png" % sequence)

        plt.figure(4)
        plt.xlabel("frame # []")
        plt.ylabel("angle [rad]")
        plt.title("KITTI Seq. %s Yaw " % sequence)
        plt.legend()
        plt.savefig(data_dir + "%s_#00_3_fig_yaw_trajectory.png" % sequence)

        plt.figure(5)
        plt.xlabel("frame # []")
        plt.ylabel("angle [rad]")
        plt.title("KITTI Sequence %s Pitch" % sequence)
        plt.legend()
        plt.savefig(data_dir + "%s_#00_4_fig_pitch_trajectory.png" % sequence)

        plt.figure(6)
        plt.xlabel("frame # []")
        plt.ylabel("angle [rad]")
        plt.title("KITTI Sequence %s Roll" % sequence)
        plt.legend()
        plt.savefig(data_dir + "%s_#00_5_fig_roll_trajectory.png" % sequence)

        print("Plot saved for sequence %s" % sequence)

    # plot the ekf states
    file_not_found = False
    ekf_state_fp = os.path.join(data_dir, "%s_ekf_states.npy" % sequence)
    fc_ground_truth_fp = os.path.join(data_dir, "%s_fc_ground_truth.npy" % sequence)
    imu_measurements_fp = os.path.join(data_dir, "%s_imu_measurements.npy" % sequence)

    ekf_state_labels = ["Delta X", "Delta Y", "Delta Z",

                        "Velocity X", "Velocity Y", "Velocity Z",

                        "Pitch wrt Gravity", "Roll wrt Gravity",

                        "Accel Bias X", "Accel Bias Y", "Accel Bias Z",

                        "Delta Yaw", "Delta Pitch", "Delta Roll",

                        "Gyro Bias Yaw", "Gyro Bias Pitch", "Gyro Bias Roll",

                        "Delta X Ground Truth", "Delta Y Ground Truth",
                        "Delta Z Ground Truth",

                        "Delta Yaw Ground Truth", "Delta Pitch Ground Truth",
                        "Delta Roll Ground Truth",

                        "IMU Rate Roll", "IMU Rate Pitch", "IMU Rate Yaw",

                        "IMU Accel X", "IMU Accel Y", "IMU Accel Z"]

    ekf_state_units = ["m", "m", "m",
                       "m/s", "m/s", "m/s",
                       "rad", "rad",
                       "m/s^2", "m/s^2", "m/s^2",
                       "rad", "rad", "rad",
                       "rad/s", "rad/s", "rad/s",
                       "m", "m", "m",
                       "rad", "rad", "rad",
                       "rad/s", "rad/s", "rad/s",
                       "m/s^2", "m/s^2", "m/s^2"]

    # print("%d = %d" % (len(ekf_state_labels), len(ekf_state_units)))
    assert (len(ekf_state_labels) == len(ekf_state_units))

    try:
        ekf_states = np.load(ekf_state_fp)
        fc_ground_truth = np.load(fc_ground_truth_fp)
        imu_measurements = np.load(imu_measurements_fp)
        ekf_states = np.concatenate([ekf_states, fc_ground_truth, imu_measurements], axis=1)
    except FileNotFoundError:
        file_not_found = True
        print("Cannot find %s" % ekf_state_fp)
        continue

    if not file_not_found:
        for j in range(0, ekf_states.shape[1]):
            plt.figure(i)
            plt.clf()
            x = np.array(range(0, ekf_states.shape[0]))[1:] / 10.0
            y = ekf_states[1:, j]

            plt.plot(x, y)

            plt.xlabel("time [s]")
            plt.ylabel("value [%s]" % ekf_state_units[j])
            plt.grid()

            plt.title("Seq. %s %s" % (sequence, ekf_state_labels[j]))
            plt.savefig(os.path.join(data_dir, "%s_#%02d_%s.png" % (sequence, j + 1, ekf_state_labels[j].lower().replace(" ", "_"))))

        print("State plot saved for sequence %s" % sequence)