from itertools import product
import itertools
import numpy as np
import torch

from minisurf.trig import Gyroid, Primitive, FisherS




def __make_data_set(trig_name_function=Gyroid(),
                    coef=[(1,1,1)], 
                    axis_chuncks=15,
                    return_type = torch.tensor):
    # gyroid = Gyroid()
    points = list(product(np.linspace(0,1,axis_chuncks), repeat=3))
    

    repeated_points = np.array(points*len(coef))
    repeated_coef = np.repeat(coef,  len(points), axis=0)

    features = np.concatenate((repeated_points, repeated_coef), axis=1)
    labels = trig_name_function.get_bunch(features)

    if return_type == torch.tensor:
        return  torch.tensor(features), torch.tensor(labels)
    # elif return_type == np.ndarray:
    #     return features, labels
    return features, labels



def __make_data_set_helper(trig_name_function=Gyroid(),
                    coef=[(1,1,1)], 
                    axis_chuncks=15,
                    return_type = torch.tensor,
                    ):
    # gyroid = Gyroid()
    features   = []
    labels = []
    for  ax, by, cz in coef:
        for x, y, z in product(np.linspace(0.0, 1, axis_chuncks), repeat=3):
            features = features + [[x,y,z, ax,by,cz]]
            labels   = labels   + [trig_name_function(x,y,z, ax,by,cz)]

    if return_type == torch.tensor:
        return  torch.tensor(features), torch.tensor(labels)
    elif return_type == np.ndarray:
        return features, labels
    return features, labels


    
def load_gyroid_sdf_dataset(coef=[(1,1,1)], 
                            axis_chuncks=15,
                            return_type = torch.tensor):
    '''this file loads (or builds as a better term) gyroid sdf dataset
    
    Parameters
    ----------
    coef : list of list (2d), list of tuples, default=[[(1,1,1)]
    meant to be coefficient of gyroid terms. The gyroid trig function
    is: `ax*sin(x)*cos(y) + by*sin(y)*cos(z) + cz*sin(z)*cos(x)`  
    and coef parameter sets ax, by, cz in this formula
    the defualt represents the regular gyroid coef=(ax,by,cz)=(1,1,1).
    You are also allowed to send a list of tuples for have a wide 
    range of possible datarows in your datasets.
        .. versionadded:: 0.01

    axis_chuncks : float, default=15
    specifies how exactly you want an axis (between 0 and 1) be chunkced for
    x,y, z values in  `ax*sin(x)*cos(y) + by*sin(y)*cos(z) + cz*sin(z)*cos(x)`.   
        .. versionadded:: 0.01
    return_type : torch.tensor, np.array, list,  defualt=torch.tensor

    Returns
    -------
    features, labels : see return_type input parameter above.
    features, and labels are the x,y,z, ax,by,cz values (features) paired with 
    sdf value t (lables) in 
    `t=ax*sin(x)*cos(y) + by*sin(y)*cos(z) + cz*sin(z)*cos(x)`

        '''
    return __make_data_set(Gyroid(),
                    coef=coef, 
                    axis_chuncks=axis_chuncks,
                    return_type = return_type)  




def load_primitive_sdf_dataset(coef=[(1,1,1)], 
                            axis_chuncks=15,
                            return_type = torch.tensor):
    '''to be added. pls copy and use gyroid one here.'''
    return __make_data_set(Primitive(),
                    coef=coef, 
                    axis_chuncks=axis_chuncks,
                    return_type = return_type)        

def load_fisher_s_sdf_dataset(coef=[(1,1,1)], 
                            axis_chuncks=15,
                            return_type = torch.tensor):
    '''to be added. pls copy and use gyroid one here.'''
    return __make_data_set(FisherS(),
                    coef=coef, 
                    axis_chuncks=axis_chuncks,
                    return_type = return_type)      


### just in case a class form is of use.
# class GyroidSDFDataSet:
#     def __init__(self):
#         self.gyroid = Gyroid()
#     def __call__(self, coef=product(np.array([1,2,]), repeat=3), axis_chuncks=15):
        
#         if isinstance(coef, itertools.product):
#             # assert(len(list(coef)[0])%3,0)
#             pass

#         elif coef.size != 3:
#             raise ValueError("Size of coef array should be 3.")
#         data   = []
#         target = []
#         for  ax, bx, cx in coef:
#             for x, y, z in product(np.linspace(0.0, 1, axis_chuncks), repeat=3):
#                 data = data + [[x,y,z, ax,bx,cx]]
#                 target = target + [self.gyroid(x,y,z, ax,bx,cx)]

#         self.features = torch.tensor(data)
#         self.labels = torch.tensor(target)
#         return self.features, self.labels


# class DiamondSDFDataSet:
#     def __init__(self):
#         pass    