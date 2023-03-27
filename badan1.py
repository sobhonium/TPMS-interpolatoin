from mpl_toolkits import mplot3d
from vedo import Text2D, Volume, show
from model.deepsdfmodel import DeepSDFModel1
import numpy as np
import torch
from itertools import product
from matplotlib.pylab import plot as plt


def visulaize_volume(model,  code=[1,1,1,  1,1,1], step_size = 15):
   
    if isinstance(code, list):
        code = np.array(code)
   
    print(code.shape)
    import joblib

    # saved scaler on featurs in preprocessing step is used here.
    saved_features_scaler = joblib.load('model/features_scaler.save') 
    if code.ndim==1:
        print('====>dim 1')
        code = code.reshape(1, -1)
    from minisurf.trig import Gyroid
    gyroid =  Gyroid()  
    sc = np.empty((code.shape[0], step_size, step_size, step_size))
    for idx, c in enumerate(code):
        
        
        
        a = np.arange(0, step_size)
        points = np.array(list(product(a, repeat=3)))
        #
        code_repeat = np.array([c]*points.shape[0])
        # print(code_repeat)
        inp = np.append(points, code_repeat, axis=1)
        # print(inp)
        inp = saved_features_scaler.transform(inp)
        # print(inp)
        print(model(torch.tensor(inp)))
        # for po, val in zip(points, gyroid(torch.tensor(inp))):
        for po in points:
            # print(po, val)
            print(po)
            sc[idx][po] = gyroid(po[0], po[1], po[2])

        print('c code:=====>: ',c, " :<===== done.")

    




    
    show_pairs = []
    for idx in range(code.shape[0]):
        vol = Volume(sc[idx])
        
        # print(sc[idx])
        vol.add_scalarbar3d()
        # print('numpy array from Volume:', vol.tonumpy().shape)

        lego = vol.legosurface(vmin=0, vmax=3) # volume of sdf( g(x,y,z) ) > 0
        lego.cmap('hot_r', vmin=0, vmax=3).add_scalarbar3d()
        
        show_pairs = show_pairs + [( lego, '*' )]

    # text1 = Text2D('', c='blue')
    # text2 = Text2D('..and its lego isosurface representation\nvmin=1, vmax=2', c='dr')
    print('code shape===========:', code.shape[0])
    show(show_pairs, N=code.shape[0], azimuth=10).close()
    print('show_paris shape=====:', len(show_pairs))

if __name__ == "__main__":


    # loading the trained model.
    model = torch.load("model/enitre_model")
    model.eval()
    
    
    code = [[1,1,1,  1,1,1],
            [0.75,0.75,1.5,  0.5,0.5,0.5], 
            [1.5,1.5,1.5, 0.5,0.5,0.5]       
    ]
    print(model(torch.tensor([0.5,0.5,0.5,  0.75,0.75,1.5,  0.5,0.5,0.5])))
    visulaize_volume(model, code=code, step_size=15, )  