"""Create a Volume from a numpy.mgrid"""
import numpy as np
from vedo import Volume, Text2D, show

X, Y, Z = 2*np.pi*np.mgrid[0:1:31j, 0:1:31j, 0:1:31j]
# Distance from the center at (15, 15, 15)
scalar_field =  2*np.sin(X)*np.cos(Y)+np.sin(Y)*np.cos(Z)+np.sin(Z)*np.cos(X)

print(type(scalar_field))
vol = Volume(scalar_field)
# print(vol)
vol.add_scalarbar3d()
print('numpy array from Volume:', vol.tonumpy().shape)

lego = vol.legosurface(vmin=0, vmax=3) # volume of gyroid > 0
lego.cmap('hot_r', vmin=0, vmax=3).add_scalarbar3d()

text1 = Text2D(__doc__, c='blue')
text2 = Text2D('..and its lego isosurface representation\nvmin=1, vmax=2', c='dr')

show([(vol,text1), (lego,text2)], N=2, azimuth=10)