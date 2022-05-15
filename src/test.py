import os
import torch
import numpy as np

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
from torch_geometric.data import InMemoryDataset, Dataset, Data
from torch_geometric.loader import DataLoader
import math

from data_util import ConTrafficGraphDataset as trafficDataset
from model import ConGAE,ConGAE_t, ConGAE_sp, deepConGAE


parser = argparse.ArgumentParser()

# model config
parser.add_argument('--model', default = 'ConGAE', help = 'Model type: ConGAE, ConGAE_t, ConGAE_sp, deepConGAE')
# files
parser.add_argument('--log_dir', default = '../log/' , help = 'directory to saved model')
# to extract corresponding model 
parser.add_argument('--polluted_training', default = False, type = bool , help = 'whether to mannually pollute small fraction of training data')
parser.add_argument('--polluted_training_frac',default = 0.1, type=float, help = 'fraction of samples to be polluted')
parser.add_argument('--polluted_training_seed',  type=int, default = 1, help = 'random seed when generating polluted training set')


# retrieve model args
arg_model = parser.parse_args()



if arg_model.polluted_training == False:
    dirName =  "../data/selected_50/"
    save_name = arg_model.model
else:
    dirName = '../data/train_synthetic_pollute{}/geopy_sp_{}/'.format(arg_model.polluted_training_seed, arg_model.polluted_training_frac)
    save_name = '{}_syn{}_{}_time'.format(arg_model.model, arg_model.polluted_training_seed, arg_model.polluted_training_frac)

    

args_path = arg_model.log_dir + save_name + '_args.pkl'
with open(args_path,'rb') as f: 
    args = pickle.load(f)
root = '../data/selected_50_pg/root/'
result_dir = args.log_dir + 'results/'


# load  data
node_X = np.load(dirName + 'node_X.npy')
node_posx = np.mean(node_X[:, :2], 1)
node_posy =  np.mean(node_X[:, 2:], 1)
node_X = torch.from_numpy(node_X).float()
tt_min, tt_max =np.load(dirName + 'tt_minmax.npy' )


# ## load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model == 'ConGAE_sp':
    model = ConGAE_sp(args.input_dim, args.node_dim1,args.node_dim2, args.dropout_p,args.adj_drop, decoder = args.decoder,  n_nodes = args.n_nodes)

    
if args.model == 'ConGAE_t':
    model = ConGAE_t(args.input_dim, args.node_dim1,args.node_dim2, args.dropout_p,args.adj_drop,decoder = args.decoder, hour_emb = args.hour_emb, week_emb = args.week_emb,n_nodes = args.n_nodes)

if args.model == 'ConGAE':
    model = ConGAE(input_feat_dim=args.input_dim, node_dim1 =args.node_dim1, node_dim2=args.node_dim2, encode_dim = args.encode_dim, dropout = args.dropout_p, adj_drop = args.adj_drop, hour_emb = args.hour_emb, week_emb = args.week_emb, n_nodes = args.n_nodes)


if args.model ==  'deepConGAE':
    model = deepConGAE(args.input_dim, hidden_list = args.hidden_list, encode_dim = args.encode_dim, decode_dim = args.decode_dim, dropout = args.dropout_p, adj_drop = args.adj_drop, hour_emb = args.hour_emb, week_emb = args.week_emb,n_nodes = args.n_nodes)
    
model.float()
# model_path = args.log_dir + args.model +"_sp" + '.pt'
model_path = args.log_dir +  save_name + '.pt' 
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
print("best model is at epoch", epoch)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)


# specify loss function
criterion = nn.MSELoss()
def calc_rmse(recon_adj, adj, tt_min, tt_max):
    adj = adj * (tt_max - tt_min) + tt_min
    recon_adj = recon_adj * (tt_max - tt_min) + tt_min
    rmse = criterion(recon_adj, adj)
    return torch.sqrt(rmse)


dirTest = '../data/test_synthetic/geopy_sp_ano_0.1/'

print ('loading test data from', dirTest)
with open(dirTest + 'name_list', 'rb') as file:
    name_list = pickle.load(file)
    
# item_d: whihc time slice each id correspond to
with open(dirTest + 'item_dict', 'rb') as file:
     item_d_test = pickle.load(file)

labels = np.load(dirTest + 'labels.npy')
        
params_test = {'batch_size': 1,
          'shuffle': False,
          'num_workers': 0}

source_dir = dirTest 
sim_test_dataset = trafficDataset(root, name_list, node_X, item_d_test, source_dir = source_dir, sim = True, labels = labels)
sim_test_loader = DataLoader(sim_test_dataset, **params_test)

model = model.to('cpu')
rmse = []
i = 0
for graph_data in sim_test_loader:
    graph_data = graph_data.to('cpu')
    model.eval()
    with torch.no_grad():                   
        recon_train = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr,\
                       graph_data.hour.clone().detach(), graph_data.week.clone().detach())
        loss = calc_rmse(recon_train, graph_data.edge_attr, tt_min, tt_max) / 60
        date = sim_test_dataset.getdatetime(i)
        i += 1
#         label = int(graph_data.label.data[0])
        rmse.append([date, loss.item()])
        
df_test = pd.DataFrame(rmse)
df_test.columns = ['date','rmse']
fpr, tpr, thresholds = roc_curve(labels, df_test['rmse'].values)
roc_auc= auc(fpr, tpr)
print(save_name, 'spatial anomaly AUC score')
print(roc_auc)
