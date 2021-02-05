'''
train and test ConGAE-fc, which uses fully connected layers for spacial info, thus use dataframe instead of graph format for input.
'''

import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split

    
import argparse
import os
import sys
import random
import torch.nn as nn
from torch.utils import data
from torch.nn import functional as F

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D   
from random import shuffle
import pickle
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader, TensorDataset
import math
from sklearn.metrics import roc_curve, auc
from torch.utils.data.sampler import SubsetRandomSampler
from torch import Tensor
from importlib import reload


from model import ConGAE_fc

parser = argparse.ArgumentParser()

# model
parser.add_argument('--model', default = 'ConGAE_fc', help = 'Model type:ConAE_fc')
# training parameters
parser.add_argument('--test',  type=bool, default = True, help = 'Directly enter test mode. If False, train the model.')
parser.add_argument('--randomseed',  type=int, default = 1)
parser.add_argument('--train_epoch', type =int, default = 150 , help = 'number of training epochs')
parser.add_argument('--lr', default = 5e-5 , help = 'learning rate')
parser.add_argument('--dropout_p', default = 0.2 , help = 'drop out rate')
parser.add_argument('--verbal', default = False, type = bool , help = 'print loss during training')
# 2-layer ConGAE parameters
parser.add_argument('--hiddem_dim1', type=int, default = 300, help = 'hidden dimension of the first fc layer')
parser.add_argument('--hiddem_dim2', type=int, default = 150, help = 'hidden dimension of the second fc layer')
parser.add_argument('--hour_emb', type=int, default = 100, help = 'hour emnbedding dimension')
parser.add_argument('--week_emb', type=int, default = 100, help = 'week emnbedding dimension')

# files
parser.add_argument('--log_dir', default = '../log/' , help = 'directory to save model')

args = parser.parse_args()
print(args)

#Reproducability 
np.random.seed(seed=args.randomseed)
random.seed(args.randomseed)
torch.manual_seed(args.randomseed)

result_dir = args.log_dir + 'results/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

train_data = '../data/Q2_transposed.pkl'


dirName = '../data/selected_50_orig/'  
tt_min, tt_max =np.load(dirName + 'tt_minmax.npy' )


def fill_na(df):
    '''interpolate NA in dataset'''
    df.interpolate(inplace=True)
    df.bfill(inplace=True)
    df.fillna(df.mean().mean(), inplace=True)

def missing_rate(df):
    '''calculate missing rate'''
    return df.isnull().sum().sum() / df.shape[0] / df.shape[1]


def select_traing(df):
    # leave out NFL days
    Q1 = df.iloc[:350]
    Q2 = df.iloc[750: ] 
    Q = pd.concat([Q1, Q2], axis=0)
    return Q

x_train = pd.read_pickle(train_data )
x_train = select_traing(x_train)
fill_na(x_train)


'''construct dataset'''
data_train = x_train.values
slices = [data_train[i:i + 1] for i in range(data_train.shape[0])]
labels = [(s.weekday(), s.hour) for s in x_train.index.droplevel(0)]
indices = np.random.permutation(len(slices))
split_point = int(0.1* len(indices))

feature_dim = slices[0].shape[1]

dataset_train = TensorDataset(Tensor(slices), Tensor(labels))

'''dataloader'''
batch_size = 20
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, drop_last=True,
                   sampler=SubsetRandomSampler(indices[:-split_point]), pin_memory=True)
val_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, drop_last=True,
                     sampler=SubsetRandomSampler(indices[-split_point:]), pin_memory=True)

'''Load model'''

model = ConGAE_fc(feature_dim,args.hiddem_dim1,args.hiddem_dim2, args.dropout_p,hour_emb = args.hour_emb, week_emb = args.week_emb)

model.float()
# print(model)

# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()

def calc_rmse(recon_adj, adj, tt_min, tt_max):
    adj = adj * (tt_max - tt_min) + tt_min
    recon_adj = recon_adj * (tt_max - tt_min) + tt_min
    rmse = criterion(recon_adj, adj)
    return torch.sqrt(rmse)


def train(epoch, train_loader ,test_loader, best_val, best_epoch, verbal = True):
    model.train()
    train_loss = 0
    loss_train = []
    loss_val = []
    for graph_data in train_loader:
        slice_data = graph_data[0]
        label_data = graph_data[1].long()
        slice_data = slice_data.to(device)
        label_data = label_data.to(device)
        optimizer.zero_grad()
        recon = model(slice_data, label_data[:,1].unsqueeze(1), label_data[:,0].unsqueeze(1))
        loss = criterion(recon, slice_data)
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
    for graph_val in val_loader:
        # evaluation
        model.eval()
        slice_data = graph_val[0]
        label_data = graph_val[1].long()
        slice_data = slice_data.to(device)
        label_data = label_data.to(device)
        with torch.no_grad():
            recon_val = model(slice_data,
                              label_data[:,1].unsqueeze(1), label_data[:,0].unsqueeze(1))
            mse_val = criterion(recon_val, slice_data)
        loss_val.append(mse_val.item())
    
    loss_train = sum(loss_train) / len(loss_train)
    loss_val = sum(loss_val) / len(loss_val)
 
     # print results
    if args.verbal and epoch % 10 == 0:
        print('Train Epoch: {}  loss: {:e}  val_loss: {:e}'.format(
            epoch, loss_train, loss_val ))
        rmse =  math.sqrt(loss_val) * (tt_max - tt_min)   # calc_rmse(recon, graph_data.edge_attr, tt_min, tt_max) 
        print('validation travel time rmse mean: {:e}'.format(rmse))
    
    #  early-stopping
    if loss_val < best_val:
        torch.save({
            'epoch' : epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
        best_val = loss_val
        best_epoch = epoch

    return  loss_train, loss_val, best_val, best_epoch

# ## Train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_track = []
val_track = []
model_path = args.log_dir + args.model + '.pt'

if not args.test:
    model = model.to(device)
    n_epochs = args.train_epoch
    start = time.time()
    best_epoch = 0
    best_val = float('inf') # for early stopping
    lr_decay_step_size = 50

    for epoch in range(1, n_epochs+1):
        train_loss, val_loss, best_val,epoch = train(epoch, train_loader, val_loader, best_val, best_epoch)
        loss_track.append(train_loss)
        val_track.append(val_loss)
        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.5 * param_group['lr']

    print("time for {} epochs: {:.3f} min".format(n_epochs, (time.time() - start)/60))

    torch.cuda.empty_cache()


    # plot learning curve
    plt.plot(np.array(loss_track), label = 'traiing')
    plt.plot(np.array(val_track), label = 'validaiton')
    plt.title("loss")
    plt.xlabel("# epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    # plt.ylim(0.4, 1)
    # plt.show()
    plt.savefig(result_dir + args.model +"_"+ '_training_curve.png')

# # ## examinne results


checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']

model = model.to('cpu')
print("best model is at epoch", epoch)

# ## test missing sensitivity

def constr_dataloader(df):
    '''construct dataset given test dataframe'''
    data_train = df.values
    slices = [data_train[i:i + 1] for i in range(data_train.shape[0])]
    labels = df.index
    indices = np.random.permutation(len(slices))

    feature_dim = slices[0].shape[1]

    dataset_train = TensorDataset(Tensor(slices), Tensor(labels))

    '''dataloader'''
    batch_size = 1
    loader = DataLoader(dataset=dataset_train, batch_size=batch_size, drop_last=True,
                        pin_memory=True)
    return loader

def cal_rmse_array(test_loader, model):
    '''# calcualte rmse for every item in test set'''
    model = model.to('cpu')
    rmse = np.zeros(len(test_loader))
    i=0
    for graph_data in test_loader:
        model.eval()
        with torch.no_grad():
            slice_data = graph_data[0]
            label_data = graph_data[1].long()
            slice_data = slice_data
            label_data = label_data
            recon = model(slice_data, label_data[:,1].unsqueeze(1), 
                          label_data[:,0].unsqueeze(1))
#             loss = criterion(recon, slice_data)
            loss = calc_rmse(recon, slice_data, tt_min, tt_max) / 60
            rmse[i] = loss.item()
            i += 1
    return rmse


def cal_auc(labels, rmse):
    '''auc given rmse'''
    fpr, tpr, thresholds = roc_curve(labels, rmse)
    roc_auc= auc(fpr, tpr)
    return roc_auc


'''s-t, ano frac'''
auc_avg = np.zeros(6)
ano_list = ['sp', 'time']
seed_list = ["", "2", '3', '4', '5']
# seed_list = [""]
for file_seed in seed_list:
    aucs = []
    data_dir = '../data/test_synthetic' + file_seed +'/'
    for ano in ano_list:
        for frac_ano in [0.05, 0.1, 0.2]:
            #read data
            x_test = pd.read_pickle(data_dir + '{}_ano_{}.pkl'.format(ano, frac_ano))
            y_test = pd.read_pickle(data_dir + '{}_labels_ano_{}.pkl'.format(ano, frac_ano))
            fill_na(x_test)
            # consttuct dataloader
            test_loader = constr_dataloader(x_test)
            #auc
            rmse = cal_rmse_array(test_loader, model)
            auc_score = cal_auc(y_test.values, rmse)
            aucs.append( auc_score) 
    auc_avg = auc_avg + np.array(aucs)
auc_avg = auc_avg / len(seed_list)


print(args.model)
print(auc_avg)

with open(result_dir + args.model+ '.pt', 'wb') as fp:
    pickle.dump(auc_avg, fp)

