'''
Sythetically pollute test data to generate spatial anomally
'''

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

parser = argparse.ArgumentParser()

# anomaly generation
parser.add_argument('--magitude', default = 0.2, type=float, help = 'magnitude of pollute percentage')
parser.add_argument('--frac_link', default = 0.5, type=float, help = 'fraction of links to be polluted')
parser.add_argument('--frac_anomaly', default = 0.3, type=float, help = '# fraction of anomaly')
parser.add_argument('--randomseed',  type=int, default = 1)

# file save
parser.add_argument('--log_dir', default = '../data/test_synthetic/', help = 'location to save synthetic data')

args = parser.parse_args()
print(args)

#Reproducability 
np.random.seed(seed=args.randomseed)
random.seed(args.randomseed)


def get_graph_geo(df):
    '''
    transform data from dataframe to geopytorch graph format
    '''
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

# read original data
df_mean = pd.read_pickle('../data/selected50_Q1_weekhour_orig.pkl')
df_std = pd.read_pickle('../data/selected50_Q1_std_orig.pkl')

tt_min, tt_max =np.load('../data/selected_50_orig/tt_minmax.npy' )
nodes = np.load('../data/nodes.npy' )
node_X = np.load('../data/selected_50_orig/node_X.npy')

with open('../data/node_dict', 'rb') as file:
     node_dict = pickle.load(file)


#Create save dir
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)
    


# ## simulate data


# create pollute list
leng =  df_mean.shape[1]
pollute_list = list(range(leng))
shuffle(pollute_list)


# original, resampled
df_resample = df_mean.copy()
df_resample[:] = np.random.normal(df_mean.values, df_std.values)
df_resample[df_resample > 1] = 1
df_resample[df_resample < 0] = 0 



'''inject spacial anomaly'''
 
# copy orig.
obs_rpca = df_resample.copy() #observation
S_rpca = df_resample.copy() # S_true
S_rpca[:] = False
 
num = int(args.frac_anomaly * leng)
for i in range(0, num):
    ind = pollute_list[i]
    time_lable = df_mean.columns[ind]
    # pollute
    mask = np.random.uniform(0, 1, df_mean.iloc[:,ind].shape) < args.frac_link
    series_sim = df_mean.iloc[:,ind]  + (np.random.uniform(-args.magitude , args.magitude,
                                                           df_mean.iloc[:,ind].shape))*mask
       
    # cap to 0-1
    series_sim[series_sim > 1] = 1
    series_sim[series_sim < 0] = 0
    
    # append to Q1 copy:  Q1_rpca
    obs_rpca[time_lable] = series_sim
    S_rpca[time_lable] = True


'''save as datafream for deeplearning models'''
test_transposed = obs_rpca.transpose()

test_transposed.to_pickle(args.log_dir + 'sp_{}_{}.pkl'.format(args.magitude, args.frac_link))  
S_rpca.all().to_pickle(args.log_dir + 'sp_labels_{}_{}.pkl'.format(args.magitude, args.frac_link))


'''save as graph structure for ConGAE model'''
save_geopy = True
dirName = args.log_dir + '/geopy_sp_{}_{}/'.format(args.magitude, args.frac_link)
df_save = obs_rpca

# save for geopy
if save_geopy:
    name_list = []
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
        name_list.append(name)
        
    # save meta    
    np.save(dirName + "tt_minmax" , [tt_min, tt_max])

    with open(dirName + 'name_list', 'wb') as fp:
        pickle.dump(name_list, fp)

    with open(dirName + 'item_dict', 'wb') as fp:
        pickle.dump(item_dict, fp)

    # save label
    labels = S_rpca.all().values
    np.save(dirName +'labels' , labels)

