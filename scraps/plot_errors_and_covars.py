import numpy as np
import matplotlib.pyplot as plt
import config

sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

for sequence in sequences:

    data_dir = "/home/cs4li/Dev/end_to_end_odometry/results/trajectory_results/epoch35_low_lr_goodresults/"
    fc_covars_path = data_dir + "fc_covars_%s.npy" % sequence
    fc_errors_path = data_dir + "fc_errors_%s.npy" % sequence

    try:
        fc_errors = trajectory = np.load(fc_errors_path)
        fc_covars = trajectory = np.load(fc_covars_path)
    except FileNotFoundError:
        print("Files for sequence %s does not exist" % sequence)
        continue

    fc_errors_abs = fc_errors
    fc_covars_3sig = np.sqrt(fc_covars) * 3

    plot_names = ["X", "Y", "Z", "Yaw", "Pitch", "Roll"]

    for i in range(0, 6):
        plt.figure(i)
        # plt.plot(fc_covars_3sig[1:, i], "b")
        # plt.plot(-fc_covars_3sig[1:, i], "b")
        # plt.plot(fc_errors_abs[1:, i] * 50, "r")
        plt.plot(fc_errors_abs[1:, i], "r")

        plt.xlabel("Frame #")

        # plt.legend()
        plt.title("Sequence %s %s Error & Covars" % (sequence, plot_names[i]))
        # plt.show()

        plt.savefig(data_dir + "fig_%s_%02d_%s_error_and_covars.png" % (sequence, i, plot_names[i]))
        plt.clf()

    print("Sequence %s done." % sequence)
