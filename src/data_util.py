import torch
from torch.utils import data
import numpy as np
from torch_geometric.data import InMemoryDataset, Dataset, Data


class trafficGraphDataset(InMemoryDataset):
    '''
    graph dataset for each time slices, py-geometric object
    '''
    def __init__(self, root, data_list, x, item_dict, source_dir = "../data/selected_50_orig/"):
        super().__init__(root)
        self.x = x   # node attributes
        self.data_list = data_list
        self.dir = source_dir  # read from directory
        self.item_dict = item_dict

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return []

    def download(self): 
        # Download to `self.raw_dir`.
        pass

    def process(self):
        pass

    def len(self):
        'Denotes the total number of samples'
        return len(self.data_list)

    def get(self, idx):
        'Generates one sample of data'
        # select sample
        name = self.data_list[idx]

        # load data and get label
        edge_index = np.load(self.dir + name + "-edgeind"+ ".npy")
        edge_attr = np.load(self.dir + name + "-edgeatt"+ ".npy")
        edge_index = torch.from_numpy(edge_index).type(torch.LongTensor)
        edge_attr = torch.from_numpy(edge_attr).float()
        graph = Data(x=self.x,  edge_index=edge_index, edge_attr= edge_attr)
        return graph

    def getdatetime(self, idx):
        'get datetime given index'
        name = self.data_list[idx]
        return self.item_dict[name]
    


class ConTrafficGraphDataset(trafficGraphDataset):
    '''
    graph dataset for time slices, py-geometric object, with context (time and date)
    '''
    def __init__(self, root, data_list, x, item_dict, source_dir = "../data/selected_50_orig/", dict_dir = '../data/selected_50_orig/item_dict', sim = False, labels = None):
        super().__init__(root, data_list, x, item_dict, source_dir)
        self.sim = sim  # if simulation, the lable is datetime or (week, hour) tuple, and there is label
        self.labels = labels
 
    def get(self, idx):
        'Fetche one sample of data and format the graph structure'
        # select sample
        name = self.data_list[idx]

        # load graph data
        edge_index = np.load(self.dir + name + "-edgeind"+ ".npy")
        edge_attr = np.load(self.dir + name + "-edgeatt"+ ".npy")
        edge_index = torch.from_numpy(edge_index).type(torch.LongTensor)
        edge_attr = torch.from_numpy(edge_attr).float()

        # load context (time)
        datetime = self.item_dict[name]

        # graph object
        if not self.sim:
            graph = Data(x=self.x,  edge_index=edge_index, edge_attr= edge_attr,\
                        hour = datetime.hour, week = datetime.weekday())
        else:
            graph = Data(x=self.x,  edge_index=edge_index, edge_attr= edge_attr,\
                        week = datetime[0] , hour = datetime[1], label = self.labels[idx])
        return graph

