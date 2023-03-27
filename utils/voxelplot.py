"""
==========================
3D voxel / volumetric plot
==========================

Demonstrates plotting 3D volumetric objects with `.Axes3D.voxels`.
"""

import matplotlib.pyplot as plt
import numpy as np

# from minisurf.trig import *
# from ..minisurf.trig import Gyroid
from ..minisurf.trig import Gyroid

# prepare some coordinates
x, y, z = np.indices((8, 8, 8))/8

# draw cuboids in the top left and bottom right corners, and a link between
# them
cube1 = (x < 3) & (y < 3) & (z < 3)
cube2 = (x >= 5) & (y >= 5) & (z >= 5)
gyroid = Gyroid()
link = gyroid(x,y,z)

# combine the objects into a single boolean array
voxelarray =  link

# set the colors of each object
colors = np.empty(voxelarray.shape, dtype=object)
colors[link] = 'red'
colors[cube1] = 'blue'
colors[cube2] = 'green'

# and plot everything
ax = plt.figure().add_subplot(projection='3d')
ax.voxels(voxelarray, facecolors=colors, edgecolor='k')

plt.show()