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
from torch_geometric.data import InMemoryDataset, Dataset, Data, DataLoader
import math

from data_util import ConTrafficGraphDataset as trafficDataset
from model import ConGAE,ConGAE_t, ConGAE_sp, deepConGAE


parser = argparse.ArgumentParser()

# model config
parser.add_argument('--model', default = 'ConGAE', help = 'Model type: ConGAE, ConGAE_t, ConGAE_sp, deepConGAE')
# files
parser.add_argument('--log_dir', default = '../log/' , help = 'directory to saved model')

# retrieve model args
arg_model = parser.parse_args()
args_path = arg_model.log_dir + arg_model.model + '_args.pkl'
with open(args_path,'rb') as f: 
    args = pickle.load(f)
root = '../data/selected_50_pg/root/'
result_dir = args.log_dir + 'results/'


# load  data
dirName =  "../data/selected_50_orig/"
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
model_path = args.log_dir + args.model + '.pt' 
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
print("best model is at epoch", epoch)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = model.to(device)


# specify loss function
criterion = nn.MSELoss()
def  calc_sample_rmse(recon_adj, adj, tt_min, tt_max):
    '''
    calculate the rmse of a given graph sample
    '''
    adj = adj * (tt_max - tt_min) + tt_min
    recon_adj = recon_adj * (tt_max - tt_min) + tt_min
    rmse = criterion(recon_adj, adj)
    return torch.sqrt(rmse)


def cal_rmse_list(model, data_loader):
    '''cal. the list rmse loss for every sample in the dataset'''
    model = model.to('cpu')
    rmse = []
    i = 0
    for graph_data in data_loader:
        graph_data = graph_data.to('cpu')
        model.eval()
        with torch.no_grad():    
            if args.model == 'ConGAE_sp':
                recon_train = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
            else:
                recon_train = model(graph_data.x, graph_data.edge_index, graph_data.edge_attr,torch.tensor(graph_data.hour), torch.tensor(graph_data.week))                  
            loss =  calc_sample_rmse(recon_train, graph_data.edge_attr, tt_min, tt_max) / 60
            i += 1
            rmse.append(loss.item())
    return rmse


'''cal spatial and temporal anomalies with differen anomaly ratio'''
def cal_aucs_st(data_dir):   
    aucs = []
    ano_types = ['sp', 'temp']
    for ano in ano_types:
        for frac_ano in [0.05, 0.1, 0.2]:
            '''# load testing set'''
            dirTest = data_dir  + '/geopy_{}_ano_{}/'.format(ano, frac_ano)

            with open(dirTest + 'name_list', 'rb') as file:
                name_list = pickle.load(file)

            with open(dirTest + 'item_dict', 'rb') as file:
                 item_d_test = pickle.load(file)

            labels = np.load(dirTest + 'labels.npy')

            params_test = {'batch_size': 1,
                      'shuffle': False,
                      'num_workers': 0}

            sim_test_dataset = trafficDataset(root, name_list, node_X, 
                                    item_d_test, source_dir = dirTest, sim = True, labels = labels)
            sim_test_loader = DataLoader(sim_test_dataset, **params_test)

            '''cal. testing loss'''
            rmse_test = cal_rmse_list(model, sim_test_loader)

            '''# auc for outlier'''
            fpr, tpr, thresholds = roc_curve(labels, rmse_test)
            roc_auc= auc(fpr, tpr)
            aucs.append(roc_auc)
    return aucs

# compute auc scores for 5 synthetic datasets with different seeds and average
auc_avg = np.zeros(6)
seed_list = ["", "2", '3', '4', '5']
for file_seed in seed_list:
    aucs = []
    data_dir = '../data/test_synthetic' + file_seed +'/'
    aucs = cal_aucs_st(data_dir)
    auc_avg = auc_avg + np.array(aucs)
auc_avg = auc_avg / len(seed_list)

print(args.model, 'sp-temp')
print(auc_avg)


with open(result_dir + args.model +"_st", 'wb') as fp:
    pickle.dump(auc_avg, fp)



'''
    calculate auc score for spatial anomalies with different anomaly magnitude
'''
def cal_aucs_sp(model, data_dir):    
    aucs = []
    for frac_ano in [0.25, 0.5]:
        for mag in [0.05, 0.1, 0.2]:
            '''# load testing set'''
            dirTest = data_dir +  'geopy_sp_{}_{}/'.format(mag, frac_ano)

            with open(dirTest + 'name_list', 'rb') as file:
                name_list = pickle.load(file)

            with open(dirTest + 'item_dict', 'rb') as file:
                 item_d_test = pickle.load(file)

            labels = np.load(dirTest + 'labels.npy')

            params_test = {'batch_size': 1,
                      'shuffle': False,
                      'num_workers': 0}

            sim_test_dataset = trafficDataset(root, name_list, node_X, 
                                    item_d_test, source_dir = dirTest, sim = True, labels = labels)
            sim_test_loader = DataLoader(sim_test_dataset, **params_test)
           
            '''cal. testing loss'''
            rmse_test = cal_rmse_list(model, sim_test_loader)
            
            '''auc for outlier'''
            fpr, tpr, thresholds = roc_curve(labels, rmse_test)
            roc_auc= auc(fpr, tpr)
            aucs.append(roc_auc)
    return aucs           

# compute auc scores for 5 synthetic datasets with different seeds and average
auc_avg = np.zeros(6)
seed_list = ["", "2", '3', '4', '5']
for file_seed in seed_list:
    aucs = []
    data_dir = '../data/test_synthetic' + file_seed +'/'
    aucs = cal_aucs_sp(model, data_dir)
    auc_avg = auc_avg + np.array(aucs)
auc_avg = auc_avg / len(seed_list)


print(args.model, 'sp')
print(auc_avg)

with open(result_dir + args.model +"_sp", 'wb') as fp:
    pickle.dump(auc_avg, fp)
    


       
