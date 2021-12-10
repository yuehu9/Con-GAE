#!/usr/bin/env python
# coding: utf-8

####
# synthetically inject anomalies into training data to test model robustness.


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import glob
import os 
import random
import math
from random import shuffle
import pickle
import argparse
import parser
from sklearn.metrics import roc_curve, auc


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument('--magitude', default = 0.1, type=float, help = 'magnitude of pollute percentage')
parser.add_argument('--frac_link', default = 0.5, type=float, help = 'fraction of links to be polluted')
parser.add_argument('--frac_anomaly', default = 0.05, type=float, help = '# fraction of anomaly')
parser.add_argument('--randomseed',  type=int, default = 1)
parser.add_argument('--pollute_type',  default = "space", help = 'chosse from space and time')

# file save
parser.add_argument('--log_dir', default = '../data/train_synthetic_pollute/', help = 'location to save synthetic data')

args = parser.parse_args()
print(args)


# In[17]:


#Create save dir
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)


# In[3]:


#Reproducability 
np.random.seed(seed=args.randomseed)
random.seed(args.randomseed)


# In[4]:


def get_graph_geo(df):
    edge_index = []
    edge_attr = []
    for i in range(len(df)):
        if ~np.isnan(df.iloc[i]):
            sid, did = df.index[i]
            sid = node_dict[sid]
            did = node_dict[did]
            edge_index.append([sid, did])
            edge_attr.append(df.iloc[i])
    edge_index = np.array(edge_index)
    edge_index = np.transpose(edge_index)
    edge_attr = np.array(edge_attr)
    return edge_index, edge_attr


# In[5]:


tt_min, tt_max =np.load('../data/selected_50/tt_minmax.npy' )
nodes = np.load('../data/nodes.npy' )
node_X = np.load('../data/selected_50/node_X.npy')


# In[6]:


with open('../data/node_dict', 'rb') as file:
     node_dict = pickle.load(file)


# In[7]:


Q1_norm = pd.read_pickle('../data/Q1_norm.pkl')


# In[8]:


# Q1_norm


# In[9]:


# create pollute list
leng =  Q1_norm.shape[1]
pollute_list = list(range(leng))
shuffle(pollute_list)


# In[10]:


# copy orig.
obs_rpca = Q1_norm.copy() #observation
S_rpca = Q1_norm.copy() # S_true
S_rpca[:] = False

# pollute
num = int(args.frac_anomaly * leng)


if args.pollute_type == 'space':
    for i in range(0, num):
        ind = pollute_list[i]
        time_lable = Q1_norm.columns[ind]
        # pollute
        mask = np.random.uniform(0, 1, Q1_norm.iloc[:,ind].shape) < args.frac_link
        series_sim = Q1_norm.iloc[:,ind]  + (np.random.uniform(0, args.magitude,
                                                               Q1_norm.iloc[:,ind].shape))*mask

        # cap to 0-1
        series_sim[series_sim > 1] = 1
        series_sim[series_sim < 0] = 0

        # append to Q1 copy:  Q1_rpca
        obs_rpca[time_lable] = series_sim
        S_rpca[time_lable] = True
else:
    # for rpca
    obs_rpca = Q1_norm.copy() #observation
    S_rpca = Q1_norm.copy() # S_true
    S_rpca[:] = False

    for i in range(0, num):
        ind = pollute_list[i]
#         time_lable = Q1_norm.columns[ind]
        serie = Q1_norm.iloc[:,ind]
        # pollute col 12 hours away
        col_fake = (ind + 12) % leng
        obs_rpca.iloc[:, col_fake] = serie
        S_rpca.iloc[:, col_fake] = True



'''save for geopy'''
save_geopy = True
# df_save = test_m
# dirName = "../data/test_mul_Q1_pl0.2_mk0.5_exm0.5/"
if args.pollute_type == 'space':
    dirName = args.log_dir + '/geopy_sp_{}/'.format(args.frac_anomaly)
else:
    dirName = args.log_dir + '/geopy_time_{}/'.format(args.frac_anomaly)

df_save = obs_rpca

# save for geopy
if save_geopy:
    sample_list = []
    item_dict = {} # a separate dictionary that links name to time label
    if not os.path.exists(dirName):
        os.mkdir(dirName)

    # save graph data        
    for ind in range(df_save.shape[1]):
        name = 'id-' + str(ind)
        time_lable = df_save.columns[ind]
        serie = df_save.iloc[:,ind]
        edge_index, edge_attr = get_graph_geo(serie)
        item_dict[name] = time_lable
        np.save(dirName + name + "-edgeind", edge_index)
        np.save(dirName + name + "-edgeatt", edge_attr)
        sample_list.append(name)
        
    # save meta    
    np.save(dirName + "tt_minmax" , [tt_min, tt_max])
    np.save(dirName + "node_X" , node_X )

    with open(dirName + 'sample_list', 'wb') as fp:
        pickle.dump(sample_list, fp)

    with open(dirName + 'item_dict', 'wb') as fp:
        pickle.dump(item_dict, fp)
    
    # save label
    labels = S_rpca.all().values
    np.save(dirName +'labels' , labels)


# In[20]:


with open(dirName + 'sample_list', 'wb') as fp:
    pickle.dump(sample_list, fp)


# In[ ]:




