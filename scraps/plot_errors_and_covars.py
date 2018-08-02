import numpy as np
import matplotlib.pyplot as plt
import config

sequence = "06"
data_dir = config.save_path + "/trajectory_results/"
fc_covars_path = data_dir + "fc_covars_%s.npy" % sequence
fc_errors_path = data_dir + "fc_errors_%s.npy" % sequence
fc_errors = trajectory = np.load(fc_errors_path)
fc_covars = trajectory = np.load(fc_covars_path)

fc_errors_abs = np.abs(fc_errors)
fc_covars_3sig = np.sqrt(fc_covars) * 3

plot_names = ["X", "Y", "Z", "Roll", "Pitch", "Yaw"]

for i in range(0, 6):
    plt.figure(i)
    plt.plot(fc_errors_abs[1:, i], "r")
    plt.plot(-fc_errors_abs[1:, i], "r")
    plt.plot(fc_covars_3sig[1:, i], "b")
    plt.plot(-fc_covars_3sig[1:, i], "b")

    plt.xlabel("Frame #")

    plt.legend()
    plt.title("Sequence %s %s Error & Covars" % (sequence, plot_names[i]))

    plt.savefig(data_dir + "fig_%s_%s_error_and_covars.png" % (sequence, plot_names[i]))

