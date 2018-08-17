import numpy as np
import os

dir_path = "/home/cs4li/Dev/KITTI/dataset/calibration/2011_09_26/2011_09_26_drive_0119_extract/oxts/data"
num_frames = 167
dat = []
data_order = 'lat, lon, alt, ' + \
             'roll, pitch, yaw, ' + \
             'vn, ve, vf, vl, vu, ' + \
             'ax, ay, az, af, al, au, ' + \
             'wx, wy, wz, wf, wl, wu, ' + \
             'pos_accuracy, vel_accuracy, ' + \
             'navstat, numsats, ' + \
             'posmode, velmode, orimode'
data_order = data_order.split(', ')
for i in range(num_frames):
    dat_path = os.path.join(dir_path, "%010d.txt" % i)
    dat.append(np.loadtxt(dat_path))

dat = np.array(dat)
accel = dat[:, 11:14]
accel_ave = np.average(np.sqrt(np.square(accel[:, 0]) + np.square(accel[:, 1]) + np.square(accel[:, 2])))
print(accel_ave)