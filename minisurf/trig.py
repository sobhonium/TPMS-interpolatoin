from numpy import sin, cos, pi

class TPMS:
    def __init__(self):
        pass
    def get_bunch(self, values):
 
        x = values[:,0]
        y = values[:,1]
        z = values[:,2]
        
        ax = values[:,3]
        cz = values[:,4]
        by = values[:,5]
        return self.__call__(x,y,z,ax,by,cz)
        
    
class Gyroid(TPMS):
    def __init__(self):
        super(Gyroid, self).__init__()
    def __call__(self, x, y, z, ax=1, by=1, cz=1):
        return ax*sin(2*pi*x)*cos(2*pi*y) + by*sin(2*pi*y)*cos(2*pi*z) + cz*sin(2*pi*z)*cos(2*pi*x)
    
 
    
class Primitive(TPMS):
    def __init__(self):
        super(Primitive, self).__init__()

    def __call__(self, x, y, z, ax=1, by=1, cz=1):
        return ax*cos(2*pi*x) + by*cos(2*pi*y) + cz*cos(2*pi*z)
        
    

    
class FisherS(TPMS):
    def __init__(self):
        super(FisherS, self).__init__()

    def __call__(self, x, y, z, ax=1, by=1, cz=1):
        return ax*cos(2*2*pi*x)*sin(2*pi*y)*cos(2*pi*z) + by*cos(2*pi*x)*cos(2*2*pi*y)*sin(2*pi*z) +cz*sin(2*pi*x)*cos(2*pi*y)*cos(2*2*pi*z)
        
