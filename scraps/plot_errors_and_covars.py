import numpy as np
import matplotlib.pyplot as plt
import config
import os
import sys

sequences = ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
# save_dir = "/home/cs4li/Dev/end_to_end_odometry/results/train_seq_20180812-16-22-19/"
save_dir = sys.argv[1]
data_dir = os.path.join(save_dir, "trajectory_results")

for sequence in sequences:

    # data_dir = config.save_path + "/trajectory_results/"

    fc_covars_path = os.path.join(data_dir, "%s_fc_covars.npy" % sequence)
    fc_errors_path = os.path.join(data_dir, "%s_fc_errors.npy" % sequence)

    try:
        fc_errors = np.load(fc_errors_path)
        fc_covars = np.load(fc_covars_path)
    except FileNotFoundError:
        print("----------")
        print(fc_errors_path)
        print(fc_covars_path)
        print("----------")
        print("Above Files for sequence %s does not exist" % sequence)
        continue

    fc_errors_abs = np.abs(fc_errors)
    fc_covars_3sig = np.sqrt(fc_covars) * 3

    plot_names = ["X", "Y", "Z", "Yaw", "Pitch", "Roll"]
    histogram_config = [(-0.2, 0.2, 50), (-0.2, 0.2, 50), (-0.2, 0.2, 50),
                        (-0.01, 0.01, 50), (-0.01, 0.01, 50), (-0.01, 0.01, 50), ]

    for i in range(0, 6):
        plt.figure(i)
        plt.plot(fc_covars_3sig[1:, i], "b")
        plt.plot(-fc_covars_3sig[1:, i], "b")
        plt.plot(fc_errors_abs[1:, i], "r")
        plt.plot(-fc_errors_abs[1:, i], "r")

        plt.xlabel("Frame #")

        # plt.legend()
        plt.title("Sequence %s %s Error & Covars" % (sequence, plot_names[i]))
        # plt.show()

        plt.savefig(os.path.join(data_dir, "fig_%s_%02d_%s_error_and_covars.png" % (sequence, i, plot_names[i])))
        plt.clf()

        plt.hist(fc_errors[1:, i], bins=np.linspace(start=np.min(fc_errors[1:, i]),
                                                    stop=np.max(fc_errors[1:, i]),
                                                    num=200))
        plt.title("Sequence %s %s Error Histogram" % (sequence, plot_names[i]))
        plt.savefig(os.path.join(data_dir, "fig_%s_%02d_%s_error_hist.png" % (sequence, i, plot_names[i])))
        plt.clf()

        plt.plot(fc_errors[1:, i], "r")
        plt.title("Sequence %s %s Error" % (sequence, plot_names[i]))
        plt.savefig(os.path.join(data_dir, "fig_%s_%02d_%s_error.png" % (sequence, i, plot_names[i])))
        plt.clf()

    print("Sequence %s done." % sequence)
