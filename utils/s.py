from vedo import dataurl, Volume, show
from vedo.applications import IsosurfaceBrowser
import numpy as np
from numpy import sin, cos, pi

x,y,z = 2*pi*np.mgrid[0:3:100j, 0:3:100j, 0:3:100j]
# Distance from the center at (15, 15, 15)

def mixed_tpms(a=1,b=1,c=1,d=0,e=0,f=0,g=0,h=0,i=0,j=0, k=0 ):
    
    scalar_field = a*sin(x)*cos(y) + b*sin(y)*cos(z) + c*sin(z)*cos(x) + d*cos(x)+e*cos(y)+f*cos(z)         +                g*sin(x)*sin(y)*sin(z)+ h*sin(x)*cos(y)*cos(z)+ i*cos(x)*sin(y)*cos(z) + j*cos(x)*cos(y)*sin(z)-k
    
    vol = Volume(scalar_field)
    # Generate the surface that contains all voxels in range [1,2]
    lego = vol.legosurface(0,5.5).add_scalarbar()
    return lego

def main(): 
    a,b,c,d,e,f,g,h,i,j, k = -1,1,1,1,1,-1,1,1,1,1,1
    lego1 = mixed_tpms(a=1,b=1,c=1,d=0,e=0,f=0,g=0,h=0,i=0,j=0, k=0 )
    lego2 = mixed_tpms(a=1,b=1,c=1,d=1,e=1,f=1,g=0,h=0,i=0,j=0, k=0 )
    lego3 = mixed_tpms(a=1,b=1,c=1,d=2,e=2,f=2,g=0,h=0,i=0,j=0, k=0 )
    lego4 = mixed_tpms(a=1,b=1,c=1,d=4,e=4,f=4,g=0,h=0,i=0,j=0, k=0 )
    lego5 = mixed_tpms(a=0,b=0,c=0,d=4,e=4,f=4,g=0,h=0,i=0,j=0, k=0 )
    lego6 = mixed_tpms(a=1,b=1,c=1,d=1,e=1,f=1,g=0,h=0,i=0,j=0, k=1.3 )

    # lego1  = cos(x)+cos(y)+cos(z)
    show_lego = [('gyroid', lego1), ('gryoid+primiv', lego2), 
                 ('gyroid112', lego3), (lego4, 'dimond0110+gyroid112'), (lego5, 'dimond0110+gyroid112'),
                 ('k added', lego6)
                 
                 ]
   

    show(show_lego, N=len(show_lego), axes=True)

if __name__=="__main__":
    main()