import torch
import torch.nn as nn
import torch.nn.functional as F

# FIXIT(SBN): dropout is not applied based on is_training var condition. 


# defining dropout_layer
def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    # In this case, all elements are dropped out
    if dropout == 1:
        return torch.zeros_like(X)
    # In this case, all elements are kept
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape) > dropout).float()
    return mask * X / (1.0 - dropout)


class DeepSDFModel1(nn.Module):
    def __init__(self, code_len=6, cordinate_dim=3, hidden_dim=40):
        super(DeepSDFModel1, self).__init__()
        self.is_training = True
        self.code_len = code_len
        self.cordinate_dim = cordinate_dim
#         self.func1 = nn.Linear(input_dim + code.shape[1], hidden_dim)
        # codelen is defined above 3+3 when we have ax,by,cz(3) and gyroid,...(3) 
        # for encoding this classes
        # inp_dim=3 as we have x,y,z
        self.func1 = nn.Linear(self.cordinate_dim + self.code_len, 
                               hidden_dim)

        self.func2 = nn.Linear(hidden_dim, hidden_dim)
        self.func3 = nn.Linear(hidden_dim, hidden_dim)
        self.func4 = nn.Linear(hidden_dim, hidden_dim)
        self.func5 = nn.Linear(hidden_dim, 1)
        
        self.outp = nn.Tanh()
        
                
                
    def forward(self, x):
#         z = torch.cat((x, code_), axis=0)
        z = nn.functional.relu(self.func1(x.float()))
#         return self.outp(self.func5(z))
            # Use dropout only when training the model
        if self.training == True:
            # Add a dropout layer after the first fully connected layer
            z = dropout_layer(z, dropout=0.080)
#         return self.outp(self.func5(z))    
        z = nn.functional.relu(self.func2(z.float()))
        
        if self.training == True:
            # Add a dropout layer after the first fully connected layer
            z = dropout_layer(z, dropout=0.10)
        
        z = nn.functional.relu(self.func3(z))
        
        if self.training == True:
            # Add a dropout layer after the first fully connected layer
            z = dropout_layer(z, dropout=0.10)
        
        z = nn.functional.relu(self.func4(z))
        
        if self.training == True:
            # Add a dropout layer after the first fully connected layer
            z = dropout_layer(z, dropout=0.10)
        
        z = self.func5(z)
#         z = self.func5(z)
        
        
        return self.outp(z)