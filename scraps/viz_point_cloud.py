"""
VISUALISE THE LIDAR DATA FROM THE KITTI DATASET

Based on the sample code from
    https://github.com/utiasSTARS/pykitti/blob/master/demos/demo_raw.py
And:
    http://stackoverflow.com/a/37863912

Contains two methods of visualizing lidar data interactively.
 - Matplotlib - very slow, and likely to crash, so only 1 out of every 100
                points are plotted.
              - Also, data looks VERY distorted due to auto scaling along
                each axis. (this could potentially be edited)
 - Mayavi     - Much faster, and looks nicer.
              - Preserves actual scale along each axes so items look
                recognizable

"""
import pykitti  # install using pip install pykitti
import os
import numpy as np

# Chose which visualization library to use:  "mayavi" or "matplotlib"
VISLIB = "mayavi"
# VISLIB = "matplotlib"

if VISLIB == "mayavi":
    # Plot using mayavi -Much faster and smoother than matplotlib
    # is built for 3D scientific data visualization and plotting
    from mayavi import mlab

    """
    I like to use python3 so i am giving installtion help for python3 (python2 might be similar):

    pip3 install mayavi

    sudo apt-get install build-essential git cmake libqt4-dev libphonon-dev python3-dev libxml2-dev libxslt1-dev qtmobility-dev libqtwebkit-dev

    sudo apt-get install python3-pyqt4 (mayavi needs pyqt4 to work)
    """



else:
    # Plot Using Matplotlib - Much slower than mayavi and visuals are not as goog as mayavi.
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()

# Raw Data directory information
basedir = '/home/ramesh/Documents/kitti_data'
date = '2011_09_26'
drive = '0005'

# Optionally, specify the frame range to load
# since we are only visualizing one frame, we will restrict what we load
# Set to None to use all the data
frame_range = range(0, 1, 1)

# Load the data
dataset = pykitti.odometry("/home/cs4li/Dev/KITTI/dataset", "00", frame=frame_range)

l = True  # to loop or not
n = 1  # keeping track of frames

# Plot only the ith frame (out of what has been loaded)
i = 3

# same as "velo = dataset.velo[i]"
for velo in dataset.velo:  # velo=[[x y z reflectance][....][....]....]
    if l:
        i = 5  # changing limit of looped frame
    if n == i or l:
        if VISLIB == "mayavi":
            fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
            mlab.points3d(
                velo[:, 0],  # x
                velo[:, 1],  # y
                velo[:, 2],  # z (height)
                velo[:, 2],  # Height data used for shading
                # velo[:, 3], # reflectance values
                mode="point",  # How to render each point {'point', 'sphere' , 'cube' }
                colormap='spectral',  # 'bone', 'copper','spectral','hsv','hot','CMRmap','Blues'

                # color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
                scale_factor=100,  # scale of the points
                line_width=10,  # Scale of the line, if any
                figure=fig,
            )
            mlab.show()
            if i == n:
                break
            n = n + 1
        else:
            # NOTE: Only 1 out of every 100 points are plotted using the matplotlib
            #       version to prevent crashing the computer
            skip = 50  # plot one in every `skip` points
            ax = fig.add_subplot(111, projection='3d')
            velo_range = range(0, velo.shape[0], skip)  # skip points to prevent crash
            ax.scatter(velo[velo_range, 0],  # x
                       velo[velo_range, 1],  # y
                       velo[velo_range, 2],  # z
                       c=velo[velo_range, 3],  # reflectance
                       cmap='gray')
            ax.set_title('Lidar scan (subsampled)')
            plt.show()

            if i == n:
                break
            n = n + 1
    else:
        n = n + 1