from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader

from minisurf.trig import Gyroid
from model.deepsdfmodel import DeepSDFModel1
from visualize import visulaize_volume
# gyroid = Gyroid()

# data   = []
# target = []
# for  ax, bx, cx in product(np.array([1,1.5,2,]), repeat=3):
#     for x, y, z in product(np.linspace(0.0, 1, 15), repeat=3):
#         # print( ax, bx, cx)
#         data = data + [[x,y,z, ax,bx,cx]]
#         target = target + [gyroid(x,y,z, ax,bx,cx)]

# features = torch.tensor(data)
# labels = torch.tensor(target)
# from dataset.tig_based_dataset import GyroidSDFDataSet


inital_code_lgn = 3  # from ax, by and cz
added_code_lgn = 3   # from gyroid, primitive, ... class 
code_len = added_code_lgn + inital_code_lgn


from dataset.tig_based_dataset import (load_gyroid_sdf_dataset,
                                       load_primitive_sdf_dataset)

# gyroid-like
coef=list(product(np.linspace(1,2, 3), repeat=3))
features1, labels1 = load_gyroid_sdf_dataset(coef=coef, axis_chuncks=10)
# the length of features is already==3+3
features1 = torch.column_stack(
                    (features1, torch.ones((features1.shape[0], added_code_lgn))) 
                    ) 

# from minisurf.trig import Gyroid
# gyroid = Gyroid()
# points_ = list(product(np.linspace(0,1,15), repeat=3))
# coef_ = list(product(np.linspace(1,2,3), repeat=3))

# points = np.array(points_*len(coef_))
# coef = np.repeat(coef_,  len(points_), axis=0)

# features1 = np.concatenate((points, coef), axis=1)
# labels1 = gyroid.get_bunch(features1)

# features1 = torch.tensor(features1)
# labels1 = torch.tensor(labels1)

# features1 = torch.column_stack(
#                     (features1, torch.rand(features1.shape[0], added_code_lgn)) 
#                     ) 

# print('featurs1 done!')


# primitive-like ---> it has 2*
coef=list(product(np.linspace(1,2, 3), repeat=3))
features2, labels2 = load_primitive_sdf_dataset(coef=coef, axis_chuncks=10)
features2 = torch.column_stack(
                    (features2, 2*torch.ones((features2.shape[0], added_code_lgn)))
                    ) 

# from minisurf.trig import Primitive
# primitive = Primitive()
# points_ = list(product(np.linspace(0,1,15), repeat=3))
# coef_ = list(product(np.linspace(1,2,3), repeat=3))

# points = np.array(points_*len(coef_))
# coef = np.repeat(coef_,  len(points_), axis=0)

# features2 = np.concatenate((points, coef), axis=1)
# labels2 = primitive.get_bunch(features2)

# features2 = torch.tensor(features2)
# labels2 = torch.tensor(labels2)

# features2 = torch.column_stack(
#                     (features2, 2*torch.rand(features2.shape[0], added_code_lgn)) 
#                     ) 

print('featurs2 done!')

# features = features1
# labels =labels1

features = torch.cat((features1, features2))
labels = torch.cat((labels1, labels2))



# print(features1.shape, features2.shape, features.shape)

# print('*'*100)
# define a data scaler here...

from sklearn.preprocessing import QuantileTransformer,StandardScaler
from sklearn.pipeline import Pipeline

print(features.shape)
features_scaler = Pipeline([('scale',StandardScaler()),
                #  ('normalizing', QuantileTransformer(output_distribution='normal')),
                ])

labels_scaler = Pipeline([('scale',StandardScaler()),
                 ('normalizing', QuantileTransformer(output_distribution='normal')),
                ])
print(features[20:40])
features = features_scaler.fit_transform(features)
features = torch.tensor(features)
print(features[20:40])
print(features.shape)
# still don't know if I should scale the target values (sdf values)
# labels = labels_scaler.fit_transform(labels.reshape(-1,1))
# labels = labels.flatten()
labels = labels/torch.max(labels)

# test tranform
# ress= [1.,  1.,  1.,  1.5, 1.5, 1.5, 0.5, 0.5, 0.5] 
# print('Transforming:-=====>', features_scaler.transform([ress]))
# meaan = torch.mean(features, axis=0)
# sttd = torch.std(features, axis=0)
# features = (features - meaan)/sttd

# print('meaan=', meaan)
# print('sttd=',sttd)
# labels = labels/torch.max(labels)


def load_array(data_arrays, batch_size, is_train=True): #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

train_x = features[:-features.shape[0]*20//100]
test_x  = features[-features.shape[0]*20//100:]
train_y = labels[:-labels.shape[0]*20//100]
test_y  = labels[-labels.shape[0]*20//100:]


batch_size = test_x.shape[0]

print(f'x train size= {train_x.shape}, test size= {test_x.shape}, batch size= {batch_size}')
print(f'y train size= {train_y.shape}, test size= {test_y.shape}, batch size= {batch_size}')
train_loader = load_array((train_x, train_y), batch_size)
test_loader  = load_array((test_x, test_y), batch_size)




cordinate_dim = 3  # x, y, z poisitons of one pixel
hidden_dim = 40 # this one is so decisive
    
model = DeepSDFModel1(cordinate_dim=3, 
              code_len=code_len, 
              hidden_dim=40,
              )




# hyperparameters
lr = 2e-2
epochs = 30
DEVICE = 'cpu'
print_step = 50

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss = nn.MSELoss()

print("Start training ...")
model.train()

for epoch in range(epochs):
    overall_loss = 0
    for batch_idx, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        y_hat = model(x.float()).float()
        ll = loss(y_hat.flatten(), y.float().flatten())
        overall_loss += ll.item()
        ll.backward()
        optimizer.step()
        
    print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx*batch_size))
    print(overall_loss)
    
print("Training Finished!!")
torch.save(model.state_dict(), "model/model.params")
torch.save(model.state_dict(), "model/model.params.pt")
torch.save(model, "model/enitre_model")

import joblib
joblib.dump(features_scaler, 'model/features_scaler.save') 

import numpy as np
from vedo import Text2D, Volume, show


# def visulaize_volume(ax=1, bx=1, cx=1, code=[1,1,1], step_size = 15):

    
#     sc = np.empty((step_size, step_size, step_size))
#     for idx, x in enumerate(np.linspace(0.0, 1, step_size)):
#         for idy, y in enumerate(np.linspace(0.0, 1, step_size)):
#             for idz, z in enumerate(np.linspace(0.0, 1, step_size)):
#                 inp = torch.tensor([x,y,z,  ax,bx,cx, code[0], code[1], code[2]])
                
#                 # scaler only applied to a matrix not a vector. damn it!
#                 inp = features_scaler.transform(inp.reshape(1,-1))
#                 inp= torch.tensor(inp)
#                 sc[idx, idy, idz]=model(inp.flatten())
#                 # if  >= 0:
#                 # sc[idx, idy, idz]=0.75
#                 # else:
#                     # sc[idx, idy, idz]=0.5    
   
#     vol = Volume(sc)
#     # print(vol.tonumpy)
#     vol.add_scalarbar3d()
#     # print('numpy array from Volume:', vol.tonumpy().shape)

#     lego = vol.legosurface(vmin=0, vmax=3) # volume of sdf( g(x,y,z) ) > 0
#     lego.cmap('hot_r', vmin=0, vmax=3).add_scalarbar3d()

#     text1 = Text2D('')
#     text2 = Text2D('lego isosurface representation', c='dr')

#     show([(vol,text1), (lego,text2)], N=2, azimuth=10)

print("Let's show")
from mpl_toolkits import mplot3d
from vedo import Text2D, Volume, show

model.eval() # deactivates dropout layer
# print(model(torch.tensor([0.5,0.5,0.5,  1.5,1.5,1.5,  1,1,1])))
# print(model(torch.tensor([0.5,0.5,0.5,  1.5,1.5,1.5,  1,1,1])))

code = [[1,1,1,  1,1,1],
        # [0.75,0.75,1.5,  0.5,0.5,0.5]       
]
# print(model(torch.tensor([0.5,0.5,0.5,  0.75,0.75,1.5,  0.5,0.5,0.5])))
visulaize_volume(model, code=code, step_size=15, ) 

# visulaize_volume(ax=1, bx=1, cx=1, step_size=20)  


