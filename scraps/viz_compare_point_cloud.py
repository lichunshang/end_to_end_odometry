import pykitti  # install using pip install pykitti
import os
import numpy as np
from mayavi import mlab

# Raw Data directory information
basedir = '/home/ramesh/Documents/kitti_data'
date = '2011_09_26'
drive = '0005'

# Load the data
dataset1 = pykitti.odometry("/home/cs4li/Dev/KITTI/dataset", "00", frame=None)
dataset2 = pykitti.raw("/home/cs4li/Dev/KITTI/dataset", "2011_10_03", "0027", frame=None)

velo1 = dataset1.get_velo(2968)
velo2 = dataset1.get_velo(2969)

fig = mlab.figure(bgcolor=(0, 0, 0), size=(640, 360))
mlab.points3d(
        velo1[:, 0],  # x
        velo1[:, 1],  # y
        velo1[:, 2],  # z (height)
        velo1[:, 2],  # Height data used for shading
        # velo[:, 3], # reflectance values
        mode="point",  # How to render each point {'point', 'sphere' , 'cube' }
        # colormap='spectral',  # 'bone', 'copper','spectral','hsv','hot','CMRmap','Blues'

        color=(1, 0, 0),     # Used a fixed (r,g,b) color instead of colormap
        scale_factor=100,  # scale of the points
        line_width=10,  # Scale of the line, if any
        figure=fig,
)
mlab.points3d(
        velo2[:, 0],  # x
        velo2[:, 1],  # y
        velo2[:, 2],  # z (height)
        velo2[:, 2],  # Height data used for shading
        # velo[:, 3], # reflectance values
        mode="point",  # How to render each point {'point', 'sphere' , 'cube' }
        colormap='spectral',  # 'bone', 'copper','spectral','hsv','hot','CMRmap','Blues'

        color=(0, 1, 0),     # Used a fixed (r,g,b) color instead of colormap
        scale_factor=100,  # scale of the points
        line_width=10,  # Scale of the line, if any
        figure=fig,
)
mlab.show()
