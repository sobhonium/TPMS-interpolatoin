from itertools import product

import joblib
import numpy as np
import torch
from vedo import Text2D, Volume, show


def plotting(model, ax=1, bx=1, cx=1, codes=[[1,1,1]], axis_chuncks = 15):
  

    # saved scaler on featurs in preprocessing step is used here.
    saved_features_scaler = joblib.load('model/features_scaler.save') 
    
    if isinstance(codes,list):
        codes = np.array(codes) 
    if codes.ndim==1:
        print('====>dim 1')
        codes = codes.reshape(1, -1)
    show_pairs = []
    for code in codes:
        sc = np.empty((axis_chuncks, axis_chuncks, axis_chuncks))
        for idx, x in enumerate(np.linspace(0.0, 1, axis_chuncks)):
            for idy, y in enumerate(np.linspace(0.0, 1, axis_chuncks)):
                for idz, z in enumerate(np.linspace(0.0, 1, axis_chuncks)):
                    inp = [[x,y,z,  ax,bx,cx, code[0], code[1], code[2]]]
                    inp = torch.tensor(saved_features_scaler.transform(inp))
                    sc[idx, idy, idz]=model(inp)
       
    
        vol = Volume(sc)
        vol.add_scalarbar3d()
        

        lego = vol.legosurface(vmin=-3, vmax=0) # volume of sdf( g(x,y,z) ) > 0
        lego.cmap('hot_r', vmin=0, vmax=3).add_scalarbar3d()


        show_pairs = show_pairs + [(lego, str(code))]

        
        print('preparing code shape#:', len(show_pairs), ',  code:', code)

    show(show_pairs, N=codes.shape[0], azimuth=10)

if __name__ == "__main__":


    # loading the trained model.
    model = torch.load("model/enitre_model")
    model.eval()
    
    # you can simply uncomment this codes and use them  
    # codes = [
        # [0.8,0.8,0.8],
        # [0.9,0.9,0.9],
        # [1.0,1.0,1.0],
        # [1.2,1.2,1.2],
        # [1.4,1.4,1.4],
        # [1.6,1.6,1.6],
        # [1.8,1.8,1.8],

        # [1,0,0],
        # [0,1,0],
        # [0,0,1],
        # [0.5,1.2,1.5],
        # [0.9,0.52,0.1],
        # [0.5,0.5,0.2],
        # [0,0.75,0.4],
        # [0.5,0,0.75]
        # ]
    
    # or you can also use this way for codes creation.
    codes = list(product(np.linspace(0, 1.2, 2), repeat=3))
    print('codes length:= ', len(codes))
    plotting(model, ax=1, bx=1, cx=1, codes=codes, axis_chuncks = 15)
