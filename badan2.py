import numpy as np
from vedo import Volume, Text2D, show

X, Y, Z = np.mgrid[:30, :30, :30]
# Distance from the center at (15, 15, 15)
scalar_field = ((X-15)**2 + (Y-15)**2 + (Z-15)**2) /225

vol = Volume(scalar_field)
vol.add_scalarbar3d()


lego = vol.legosurface(vmin=1, vmax=2)
lego.cmap('hot_r', vmin=1, vmax=2).add_scalarbar3d()



show([('1',lego), ('2',lego), ('3', lego)], N=3, azimuth=10).close()