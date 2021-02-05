import os

import torch
import numpy as np
from torch_geometric.data import Data

from sklearn.metrics import roc_curve, auc
import argparse
import os
import sys
import random
import torch.nn as nn
from torch.utils import data

import pandas as pd
import matplotlib.pyplot as plt 
from random import shuffle
import pickle
import torchvision.transforms as transforms
import time
from torch_geometric.data import InMemoryDataset, Dataset, Data, DataLoader
import math

from data_util import ConTrafficGraphDataset as trafficDataset
from model import ConGAE,ConGAE_t, ConGAE_sp, deepConGAE

parser = argparse.ArgumentParser()

# model
parser.add_argument('--model', default = 'ConGAE', help = 'Model type: ConGAE, ConGAE_t, ConGAE_sp, deepConGAE')
# training parameters
parser.add_argument('--randomseed',  type=int, default = 1)
parser.add_argument('--train_epoch', type =int, default = 150 , help = 'number of training epochs')
parser.add_argument('--lr', default = 5e-5 , help = 'learning rate')
parser.add_argument('--dropout_p', default = 0.2 , help = 'drop out rate')
parser.add_argument('--adj_drop', default = 0.2 , help = 'edge dropout rate')
parser.add_argument('--verbal', default = False, type = bool , help = 'print loss during training')
# 2-layer ConGAE parameters
parser.add_argument('--input_dim', type=int, default = 4, help = 'input feature dimension')
parser.add_argument('--n_nodes', type=int, default = 50, help = 'total number of nodes in the graph')
parser.add_argument('--node_dim1', type=int, default = 300, help = 'node embedding dimension of the first GCN layer')
parser.add_argument('--node_dim2', type=int, default = 150, help = 'node embedding dimension of the second GCN layer')
parser.add_argument('--encode_dim', type=int, default = 150, help = 'final graph embedding dimension of the Con-GAE encoder')
parser.add_argument('--hour_emb', type=int, default = 100, help = 'hour emnbedding dimension')
parser.add_argument('--week_emb', type=int, default = 100, help = 'week emnbedding dimension')
parser.add_argument('--decoder', type=str, default = 'concatDec', help = 'decoder type:concatDec, bilinearDec')
# deepConGAE parameters
parser.add_argument('--hidden_list', nargs="*", type=int, default = [300, 150], help = 'the node embedding dimension of each layer of GCN')
parser.add_argument('--decode_dim', type=int, default = 150, help = 'the node embedding dimension at decoding')

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


# ## load data

dirName =  "../data/selected_50_orig/"
with open(dirName + 'partition_dict', 'rb') as file:
    partition = pickle.load(file)
    
# item_d: whihc time slice each id correspond to
with open(dirName + 'item_dict', 'rb') as file:
     item_d = pickle.load(file)

node_X = np.load(dirName + 'node_X.npy')
node_posx = np.mean(node_X[:, :2], 1)
node_posy =  np.mean(node_X[:, 2:], 1)

node_X = torch.from_numpy(node_X).float()
tt_min, tt_max =np.load(dirName + 'tt_minmax.npy' )


start_time = 0+24*23
end_time = 23+24*27

# reset partition
all_data = partition['train'] + partition['val']
partition_test = all_data[350:750] # includes NFL, 400 points, 30% are NFL
partition_val = all_data[:150]  
partition_train = all_data[150:350] + all_data[750:] # the rest


source_dir = dirName # full sample size (~2000)

# Parameters
params = {'batch_size': 10,
          'shuffle': True,
          'num_workers': 0}

params_val = {'batch_size': 10,
          'shuffle': False,
          'num_workers': 0}

root = '../data/selected_50_pg/root/'
# data loaders
train_dataset = trafficDataset(root, partition_train, node_X,  item_d, source_dir  )
test_dataset = trafficDataset(root, partition_test, node_X, item_d, source_dir)
val_dataset = trafficDataset(root, partition_val, node_X, item_d, source_dir)

train_loader = DataLoader(train_dataset, **params)
test_loader = DataLoader(test_dataset,**params_val )
val_loader = DataLoader(val_dataset,**params_val )


# ## load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model == 'ConGAE_sp':
    model = ConGAE_sp(args.input_dim, args.node_dim1,args.node_dim2, args.dropout_p,args.adj_drop, decoder = args.decoder,  n_nodes = args.n_nodes)

    
if args.model == 'ConGAE_t':
    model = ConGAE_t(args.input_dim, args.node_dim1,args.node_dim2, args.dropout_p,args.adj_drop,decoder = args.decoder, hour_emb = args.hour_emb, week_emb = args.week_emb,n_nodes = args.n_nodes)

if args.model == 'ConGAE':
    model = ConGAE(input_feat_dim=args.input_dim, node_dim1 =args.node_dim1, node_dim2=args.node_dim2, encode_dim = args.encode_dim ,hour_emb = args.hour_emb, week_emb = args.week_emb, n_nodes = args.n_nodes)


if args.model ==  'deepConGAE':
    model = deepConGAE(args.input_dim, hidden_list = args.hidden_list, encode_dim = args.encode_dim,decode_dim = args.decode_dim, dropout = args.dropout_p, adj_drop = args.adj_drop, hour_emb = args.hour_emb, week_emb = args.week_emb,n_nodes = args.n_nodes)
    
model.float()


# specify optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.MSELoss()


def calc_rmse(recon_adj, adj, tt_min, tt_max):
    adj = adj * (tt_max - tt_min) + tt_min
    recon_adj = recon_adj * (tt_max - tt_min) + tt_min
    rmse = criterion(recon_adj, adj)
    return torch.sqrt(rmse)


def train(epoch, train_loader ,test_loader, best_val):
    model.train()
    train_loss = 0
    loss_train = []
    loss_val = []
    for graph_data in train_loader:
        graph_data = graph_data.to(device)
        optimizer.zero_grad()
        if args.model == 'ConGAE_sp':
            recon = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
        else:
            recon = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr,graph_data.hour, graph_data.week)
        loss = criterion(recon, graph_data.edge_attr)
        loss.backward()
        optimizer.step()
        loss_train.append(loss.item())
    for graph_val in val_loader:
        # evaluation
        model.eval()
        graph_val = graph_val.to(device)
        with torch.no_grad():
            if args.model == 'ConGAE_sp':
                recon_val = model(graph_val.x, graph_val.edge_index, graph_val.edge_attr)
            else:
                recon_val = model(graph_val.x, graph_val.edge_index, graph_val.edge_attr,                            graph_val.hour, graph_val.week)
            mse_val = criterion(recon_val, graph_val.edge_attr)
        loss_val.append(mse_val.item())
    
    loss_train = sum(loss_train) / len(loss_train)
    loss_val = sum(loss_val) / len(loss_val)
 
    # print results
    if args.verbal and epoch % 15 == 0:
        print('Train Epoch: {}  loss: {:e}  val_loss: {:e}'.format(
            epoch, loss_train, loss_val ))
        rmse =  math.sqrt(loss_val) * (tt_max - tt_min)
        print('validation travel time rmse mean: {:e}'.format(rmse))
    
    #  early-stopping
    if loss_val < best_val:
        torch.save({
            'epoch' : epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, model_path)
        best_val = loss_val

    return  loss_train, loss_val, best_val


# ## Train

loss_track = []
val_track = []

model = model.to(device)
n_epochs = args.train_epoch
start = time.time()
best_val = float('inf') # for early stopping
model_path = args.log_dir + args.model  + '.pt'
lr_decay_step_size = 100

for epoch in range(1, n_epochs+1):
    train_loss, val_loss, best_val = train(epoch, train_loader, val_loader, best_val)
    loss_track.append(train_loss)
    val_track.append(val_loss)
    if epoch % lr_decay_step_size == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']
    
print("time for {} epochs: {:.3f} min".format(n_epochs, (time.time() - start)/60))


# plot learning curve
plt.plot(np.array(loss_track), label = 'traiing')
plt.plot(np.array(val_track), label = 'validaiton')
plt.title("loss")
plt.xlabel("# epoch")
plt.ylabel("MSE loss")
plt.legend()
# plt.ylim(0.4, 1)
# plt.show()
plt.savefig(result_dir + args.model +"_training_curve.png")


# save args config
with open(args.log_dir + args.model + '_args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
    