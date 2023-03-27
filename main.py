from itertools import product

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from torch.utils import data
from torch.utils.data import DataLoader
from vedo import Text2D, Volume, show

from dataset.tig_based_dataset import (load_fisher_s_sdf_dataset,
                                       load_gyroid_sdf_dataset,
                                       load_primitive_sdf_dataset)
from model.deepsdfmodel import DeepSDFModel1

#========================================================
# features and labels and codes lengths (dataset creation)

inital_code_lgn = 3  # from ax, by and cz
added_code_lgn = 3   # from gyroid, primitive, ... class 
code_len = added_code_lgn + inital_code_lgn



# gyroid-like
coef=list(product(np.linspace(1,2, 3), repeat=3))
features1, labels1 = load_gyroid_sdf_dataset(coef=coef, axis_chuncks=20)
# the length of features is already==3+3
features1 = torch.column_stack(
                    (features1, torch.zeros((features1.shape[0], added_code_lgn))) 
                    ) 
features1[:,6] = 1
# primitive-like ---> it has 2*
coef=list(product(np.linspace(1,2, 3), repeat=3))
features2, labels2 = load_primitive_sdf_dataset(coef=coef, axis_chuncks=20)
features2 = torch.column_stack(
                    (features2, 2*torch.zeros((features2.shape[0], added_code_lgn)))
                    ) 
features2[:,7] = 1

# FisherS-like ---> it has 2*
coef=list(product(np.linspace(1,2, 3), repeat=3))
features3, labels3 = load_fisher_s_sdf_dataset(coef=coef, axis_chuncks=20)
features3 = torch.column_stack(
                    (features3, 3*torch.zeros((features3.shape[0], added_code_lgn)))
                    ) 
features3[:,8] = 1


features = torch.cat((features1, features2))
labels = torch.cat((labels1, labels2))

features = torch.cat((features, features3))
labels = torch.cat  ((labels, labels3))
#========================================================

#========================================================
# scaling data features and labels
features_scaler = Pipeline([('scale',StandardScaler()),
                #  ('normalizing', QuantileTransformer(output_distribution='normal')),
                ])

labels_scaler = Pipeline([('scale',StandardScaler()),
                 ('normalizing', QuantileTransformer(output_distribution='normal')),
                ])

features = features_scaler.fit_transform(features)
features = torch.tensor(features)

# meaan = torch.mean(features, axis=0)
# sttd = torch.std(features, axis=0)
# features = (features - meaan)/sttd

labels = labels/torch.max(labels)
#========================================================



#========================================================
# splitting the dataset for train and test
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
#========================================================


#========================================================
# instantiating a model
cordinate_dim = 3  # x, y, z poisitons of one pixel
hidden_dim = 40 # this one is so decisive
    
model = DeepSDFModel1(cordinate_dim=3, 
              code_len=code_len, 
              hidden_dim=40,
              )
#========================================================

#========================================================
# hyperparameters setting
lr = 2e-2
epochs = 30
DEVICE = 'cpu'
print_step = 50
#========================================================

#========================================================
# training
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
#========================================================

#========================================================
# Saving model and its parameters.
torch.save(model.state_dict(), "model/model.params")
torch.save(model.state_dict(), "model/model.params.pt")
torch.save(model, "model/enitre_model")

joblib.dump(features_scaler, 'model/features_scaler.save') 
#========================================================





#========================================================
# plot to see the accuracy
model.eval() # deactivates dropout layer
# uncomment if you still doubt the dropout to check if it truly is deterministic.
# print(model(torch.tensor([0.5,0.5,0.5,  1.5,1.5,1.5,  1,1,1])))
# print(model(torch.tensor([0.5,0.5,0.5,  1.5,1.5,1.5,  1,1,1])))



# some random codes to test the training model. Some of these codes are meant to be
codes = [
        [0.8,0.8,0.8],
        [0.9,0.9,0.9],
        [1.0,1.0,1.0],
        ]

from plotian import plotting

print("Let's show")
plotting(model=model, codes=codes)
#========================================================
