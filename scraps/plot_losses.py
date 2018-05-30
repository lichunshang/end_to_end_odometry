import numpy as np
import matplotlib.pyplot as plt
import tools
import config
import os

data_dir = os.path.join(config.save_path, "trajectory_results")

losses_to_overlay = [
    (data_dir + "run_results_trajectory_results_1-tag-se3_losses.csv", {"linewidth": 1.0, "color": "r"}, "Training Loss"),
    (data_dir + "run_results_trajectory_results_1-tag-se3_losses_val.csv", {"linewidth": 1.0, "color": "b"}, "Validation Loss")
]

plt.figure(1)

loss_to_draw = losses_to_overlay[0]
data = np.loadtxt(loss_to_draw[0], skiprows=1, delimiter=",")
x = data[:, 1] / 797
y = data[:, 2]
y = tools.smooth(y, window_len=2000, window="hanning")
plt.plot(x, y[0: len(x)], **loss_to_draw[1], label=loss_to_draw[2])

loss_to_draw = losses_to_overlay[1]
data = np.loadtxt(loss_to_draw[0], skiprows=1, delimiter=",")
x = data[:, 1] / 24
y = data[:, 2]
y = tools.smooth(y, window_len=100, window="hanning")
plt.plot(x, y[0: len(x)], **loss_to_draw[1], label=loss_to_draw[2])

plt.xlabel("epochs")
plt.ylabel("SE(3) losses")
plt.ylim(0, 0.1)
plt.xlim(0, 151)
plt.title("SE(3) Training and Validation Losses")
plt.legend()
plt.savefig(data_dir + "fig_losses.png")
plt.show()
