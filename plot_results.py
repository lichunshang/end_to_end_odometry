import numpy as np
import sys
import os.path
import matplotlib.pyplot as plt

filenames = ['fc.npy',
             'fc_xyz.npy',
             'fc_ypr.npy',
             'se3.npy',
             'se3_quat.npy',
             'se3_val.npy',
             'se3_xyz.npy',
             'total.npy']

if len(sys.argv) is not 2:
    raise ValueError('Need to provide results directory')

results_path = sys.argv[1]

found_files = []

for filename in filenames:
    found_files.append(os.path.isfile(results_path + filename))

if not any(found_files):
    raise ValueError('No result files in directory')

for i in range(len(filenames)):
    if found_files[i]:
        plt.figure()
        data = np.load(results_path + filenames[i])
        data = data[~(data==0).all(1)]
        data = np.mean(data, axis=1)
        plt.semilogy(data)
        plt.title(filenames[i])

plt.show()
