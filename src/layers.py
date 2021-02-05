import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class bilinearDec(Module):
    '''
    predict adj matrix (edges) by A = z * Q * z' + b, z is node embedding (N, h), Q leranable (h, h)
    https://pytorch.org/docs/stable/nn.html#bilinear
    '''

    def __init__(self, embed_features):
        super().__init__()
        self.embed_features = embed_features
        self.bl = torch.nn.Bilinear(self.embed_features, self.embed_features, 1)
    def forward(self, z, edge_index):
        '''
        output: k*1, k : # of edges
        '''
        y = self.bl(z[edge_index[0]], z[edge_index[1]])
        y = y.flatten()
        return y
    

class concatDec(Module):
    '''
    predict adj matrix (edges) by MLP, whose input is 
    concatenate([z(sid), z(did)])
    '''
    def __init__(self, embed_features, hid_dim, drop_rate = 0.2):
        super(concatDec, self).__init__()
        self.fc1 = torch.nn.Linear(embed_features * 2, hid_dim)
        self.fc2 = torch.nn.Linear(hid_dim , 1)
        self.feature_drop = torch.nn.Dropout(drop_rate)

    def forward(self, z, edge_index):
        '''
        output: k*1, k : # of edges
        '''
        z_cat = torch.cat((z[edge_index[0]], z[edge_index[1]]), 1)
        z_cat = self.feature_drop(z_cat)
        y = self.fc1(z_cat)
        y = F.relu(y)
        y = self.fc2(y).flatten()
        return y


