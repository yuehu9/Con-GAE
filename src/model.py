import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GCNConv, SAGEConv, LEConv, GraphConv, ARMAConv, SGConv, TAGConv, TransformerConv
from torch_geometric.utils import dropout_adj

from layers import bilinearDec, concatDec

 
class ConGAE(torch.nn.Module): 
    '''
    spatial-temporal graph GAE model, with 2 layers of GCN at encoding step.
    '''
    def __init__(self, input_feat_dim, node_dim1, node_dim2, encode_dim ,dropout= 0.2, adj_drop = 0.2,encoder = 'GraphConv',  decoder = 'concatDec', sigmoid = True, n_nodes = 50, hour_emb = 100, week_emb = 100, ARMAConv_num_stacks = 1, ARMAConv_num_layers = 1, TransformerConv_heads = 1):
        '''
        input_feat_dim = input feature dimension
        node_dim1, node_dim2 - the node embedding dimensions of the two layers of GCN.
        encode_dim - final graph node embedding dimension
        adj_drop - graph edge dropout rate
        decoder - choose from 'bilinearDec' and 'concatDec'(an MLP decoder)
        sigmoid - whether edge prediction output go through a sigmoid operator to (0,1)
        n_nodes - total number of nodes in the graph
        hour_emb, week_emb - the dimension of hour and week embeddings
        '''
        super().__init__()
        self.n_nodes =  n_nodes
        self.dropout = dropout
        self.adj_drop = adj_drop
        self.sigmoid = sigmoid
        self.node_dim2 = node_dim2
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.encoder = encoder
        # encode
        if encoder == 'SAGE':
            self.conv1 = SAGEConv(input_feat_dim, node_dim1) #, concat = True)
            self.conv2 = SAGEConv(node_dim1 , node_dim2) #, concat = True)
        elif encoder == 'GraphConv':
            self.conv1 = GraphConv(input_feat_dim, node_dim1, aggr = 'mean')
            self.conv2 = GraphConv(node_dim1 , node_dim2, aggr = 'mean')
        elif encoder == 'TAGConv':
            self.conv1 = TAGConv(input_feat_dim, node_dim1)
            self.conv2 = TAGConv(node_dim1 , node_dim2)
        elif encoder == 'SGConv':
            self.conv1 = SGConv(input_feat_dim, node_dim1)
            self.conv2 = SGConv(node_dim1 , node_dim2)
        elif encoder == 'ARMAConv':
            self.conv1 = ARMAConv(input_feat_dim, node_dim1, num_stacks = ARMAConv_num_stacks, num_layers = ARMAConv_num_layers)
            self.conv2 = ARMAConv(node_dim1 , node_dim2, num_stacks = ARMAConv_num_stacks, num_layers = ARMAConv_num_layers)
        elif encoder == 'TransformerConv':
            self.conv1 = TransformerConv(input_feat_dim, node_dim1, heads = TransformerConv_heads, edge_dim = 1)
            self.conv2 = TransformerConv(node_dim1*TransformerConv_heads , node_dim2, heads = 1, edge_dim = 1)
        else:
            raise NotImplementedError
        self.hour_embedding = nn.Embedding(24, hour_emb)
        self.week_embedding = nn.Embedding(7, week_emb)
        self.fc = torch.nn.Linear(n_nodes * node_dim2+hour_emb+week_emb, encode_dim)
        # decode
        self.fc2 =  torch.nn.Linear(encode_dim+hour_emb+week_emb, n_nodes * node_dim2)
        if decoder == 'bilinearDec':
            self.decoder = bilinearDec(node_dim2)
        elif decoder == 'concatDec':
            self.decoder = concatDec(node_dim2, node_dim1, dropout)
        else:
            raise NotImplementedError

    def encode(self, x, edge_index, edge_attr,hour, week):
        '''
        x - node input features
        hour - int between 0-23 for 24 hours of day
        week - int between 0-6 for 7 days of week
        '''
        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.adj_drop, num_nodes=len(x), training=self.training)
        #graph conv
        if self.encoder == 'SAGE':
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.dropout_layer(x)
            x = self.conv2(x, edge_index)
        elif self.encoder == 'TransformerConv':
            x = self.conv1(x, edge_index, edge_attr = edge_attr)
            x = F.relu(x)
            x = self.dropout_layer(x)
            x = self.conv2(x, edge_index, edge_attr = edge_attr)
        else:
            x = self.conv1(x, edge_index, edge_weight = edge_attr)
            x = F.relu(x)
            x = self.dropout_layer(x)
            x = self.conv2(x, edge_index, edge_weight = edge_attr)
        x = F.relu(x)
        #stack node embeddings and time embeddings
        z = x.view(-1, self.n_nodes * self.node_dim2)   # (batch_size , #nodes*node_dim2 )       
        z = self.concat_time(z, hour, week) # (batch_size , #nodes*node_dim2 +hour_dim + week_dim )     
        z = self.dropout_layer(z)
        z = self.fc(z)  #  (batch_size, encode_dim)   
        return z

    def decode(self, z, edge_index, hour,week):
        z = self.concat_time(z, hour, week) # concat time embeddings
        z = self.fc2(z)     # (batch_size , #nodes*node_dim2 )
        z = F.relu(z)
        z = z.view(-1, self.node_dim2)    #(#nodes*batch_size, node_dim2)
        edge_attr_pred = self.decoder(z, edge_index)
        if self.sigmoid:
            edge_attr_pred = torch.sigmoid(edge_attr_pred)
        return edge_attr_pred
    
    def forward(self, x, edge_index, edge_attr, hour, week):
        z = self.encode(x, edge_index, edge_attr, hour, week)
        adj = self.decode(z, edge_index,hour, week)
        return adj   

    def concat_time(self, z, hour, week):
        '''concatenate embeddings with hour and week'''
        emb_h = self.hour_embedding(hour)
        emb_w = self.week_embedding(week)
        z = torch.cat((z,emb_h, emb_w), dim=-1)
        return z

    
    
###########################################################################
############## Ablation study model variants ##############################
###########################################################################


class deepConGAE(torch.nn.Module):
    '''
    encoder-decoder for edge prediction, with multi GCN layers for encoder
    '''
    def __init__(self, input_feat_dim, hidden_list = [300, 150], encode_dim = 100, decode_dim = 100, dropout = 0.2, adj_drop = 0.2, sigmoid = True, n_nodes = 50, hour_emb = 200, week_emb = 200, decoder = 'concatDec'):
        '''
        hidden_list - the node embedding dimension of each layer of GCN
        encode_dim - final graph node embedding dimension
        decode_dim - the node embedding dimension at decoding
        '''
        # encodes. The last dimension of hidden_list is encoding dimmension
        super().__init__()
        self.n_nodes =  n_nodes
        self.dropout = dropout
        self.adj_drop = adj_drop
        self.dropout_layer = torch.nn.Dropout(dropout)
        self.hidden_list = hidden_list
        self.decode_dim = decode_dim
        self.sigmoid = sigmoid
        # encode
        self.GNNs = torch.nn.ModuleList()
        self.GNNs.append( GraphConv(input_feat_dim, hidden_list[0]))
        for i in range(len(hidden_list) - 1):
            self.GNNs.append(GraphConv(hidden_list[i], hidden_list[i+1]))

        self.hour_embedding = nn.Embedding(24, hour_emb)
        self.week_embedding = nn.Embedding(7, week_emb)
        self.fc = torch.nn.Linear(n_nodes *hidden_list[-1]+hour_emb+week_emb, encode_dim)
        # decode
        self.fc2 =  torch.nn.Linear(encode_dim+hour_emb+week_emb, n_nodes * decode_dim)
        if decoder == 'bilinearDec':
            self.decoder = bilinearDec(decode_dim)
        elif decoder == 'concatDec':
            self.decoder = concatDec(decode_dim, hidden_list[0], dropout)
        else:
            raise NotImplementedError        

    def encode(self, x, edge_index, edge_attr, hour,week):
        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.adj_drop, num_nodes=len(x), training=self.training)
        for conv in self.GNNs:
            x = F.relu(conv(x, edge_index, edge_weight = edge_attr))
            x = self.dropout_layer(x)
        z = x.view(-1, self.n_nodes * self.hidden_list[-1])
        z = self.concat_time(z, hour, week)
        z = self.dropout_layer(z)
        z = self.fc(z)    # (batch_size , node_dim )
        return z

    def decode(self, z, edge_index, hour,week):
        z = self.concat_time(z, hour, week) # concatenate hour and week
        z = F.relu(self.fc2(z))   # (batch_size, #nodes*node_dim )
        z = z.view(-1, self.decode_dim)
        edge_attr_pred = self.decoder(z, edge_index)
        if self.sigmoid:
            edge_attr_pred = torch.sigmoid(edge_attr_pred)
        return edge_attr_pred

    def forward(self, x, edge_index, edge_attr, hour, week):
        z = self.encode(x, edge_index, edge_attr, hour, week)
        adj = self.decode(z, edge_index,hour, week)
        return adj

    def concat_time(self, z, hour, week):
        # concatenate hour and week
        emb_h = self.hour_embedding(hour)
        emb_w = self.week_embedding(week)
        z = torch.cat((z,emb_h, emb_w), dim=-1)
        return z


class ConGAE_fc(torch.nn.Module):
    '''
    use fully connected layers instead of GCN for spatial info
    '''
    def __init__(self, input_feat_dim, node_dim1, node_dim2, dropout= 0.2, sigmoid = True, n_nodes = 50, hour_emb = 24, week_emb = 7):
        super().__init__()
        # encode
        self.sp_fc1 = torch.nn.Linear(input_feat_dim, node_dim1*n_nodes)
        self.sp_fc2 = torch.nn.Linear(node_dim1*n_nodes, n_nodes * node_dim2)
        self.hour_embedding = nn.Embedding(24, hour_emb)
        self.week_embedding = nn.Embedding(7, week_emb)
        self.fc = torch.nn.Linear(n_nodes * node_dim2+hour_emb+week_emb, node_dim2)
        self.fc_dropout = torch.nn.Dropout(dropout)
        #decode
        self.de_fc1 = torch.nn.Linear(node_dim2+hour_emb+week_emb, n_nodes * node_dim2)
        self.de_fc2 = torch.nn.Linear(n_nodes * node_dim2, input_feat_dim)
        self.sigmoid = sigmoid
       
    def encode(self, x, hour, week):
        x = self.sp_fc1(x)
        x = F.relu(x)
        x = self.fc_dropout(x)
        z = self.sp_fc2(x)
        z = F.relu(z)
        z = self.concat_time(z, hour, week)     
        z = self.fc_dropout(z)
        z = self.fc(z)
        return z

    def decode(self, z, hour,week):
        z = self.concat_time(z, hour, week) 
        z = self.de_fc1(z)    
        z = F.relu(z)
        edge_attr_pred = self.de_fc2(z)
        if self.sigmoid:
            edge_attr_pred = torch.sigmoid(edge_attr_pred)
        return edge_attr_pred
    
    def forward(self, x, hour, week):
        z = self.encode(x, hour, week)
        adj = self.decode(z, hour, week)
        return adj   

    def concat_time(self, z, hour, week):
        # concatenate hour and week
        emb_h = self.hour_embedding(hour)
        emb_w = self.week_embedding(week)
        z = torch.cat((z,emb_h, emb_w), dim=-1)
        return z

class ConGAE_sp(torch.nn.Module):
    '''
    only take in graph info via GCN, no time info
    '''
    def __init__(self, input_feat_dim, node_dim1, node_dim2, dropout= 0.2, adj_drop = 0.2, decoder = 'concatDec', sigmoid = True, n_nodes = 50):
        super().__init__()
        self.sigmoid = sigmoid
        self.adj_drop = adj_drop
        self.n_nodes =  n_nodes
        self.node_dim2 = node_dim2
        self.dropout_layer = torch.nn.Dropout(dropout)
        # encode
        self.conv1 = GraphConv(input_feat_dim, node_dim1)
        self.conv2 = GraphConv(node_dim1 , node_dim2)
        self.fc =  torch.nn.Linear(n_nodes * node_dim2, node_dim2)
        # decode
        self.fc2 =  torch.nn.Linear(node_dim2, n_nodes * node_dim2)         
        if decoder == 'bilinearDec':
            self.decoder = bilinearDec(node_dim2)
        elif decoder == 'concatDec':
            self.decoder = concatDec(node_dim2, node_dim1, dropout)
        else:
            raise NotImplementedError
    def encode(self, x, edge_index, edge_attr):
        edge_index, edge_attr = dropout_adj(edge_index, edge_attr, p=self.adj_drop, num_nodes=len(x), training=self.training)
        x = self.conv1(x, edge_index, edge_weight = edge_attr)
        x = F.relu(x)
        x = self.dropout_layer(x)
        z = self.conv2(x, edge_index, edge_weight = edge_attr)
        z = z.view(-1, self.n_nodes * self.node_dim2)   # (batch_size , #nodes*node_dim2 )
        z = F.relu(z)
        z = self.dropout_layer(z)
        z = self.fc(z)    # (batch_size , node_dim2 )
        return z

    def decode(self, z, edge_index):
        z = self.fc2(z)     # (batch_size_embd , #nodes*node_dim2 )
        z = F.relu(z)
        z = z.view(-1, self.node_dim2)    #(#nodes*batch_size, node_dim2)
        edge_attr_pred = self.decoder(z, edge_index)
        if self.sigmoid:
            edge_attr_pred = torch.sigmoid(edge_attr_pred)
        return edge_attr_pred
    
    def forward(self, x, edge_index, edge_attr):
        z = self.encode(x, edge_index, edge_attr)
        adj = self.decode(z, edge_index)
        return adj  

class ConGAE_t(ConGAE_sp):
    '''Only take in on hour and week info , no graph info'''
    def __init__(self, input_feat_dim, node_dim1, node_dim2, dropout= 0.2, adj_drop = 0.2, decoder = 'concatDec', sigmoid = True, n_nodes = 50, hour_emb = 100, week_emb = 100):
        super().__init__(input_feat_dim, node_dim1, node_dim2, dropout, adj_drop, decoder, sigmoid, n_nodes)
        self.fc = torch.nn.Linear(hour_emb+week_emb, node_dim2)
        self.fc2 =  torch.nn.Linear(node_dim2+hour_emb+week_emb, n_nodes * node_dim2)
        # use embedding. encode:
        self.hour_embedding = nn.Embedding(24, hour_emb)
        self.week_embedding = nn.Embedding(7, week_emb)

    def encode(self, x, edge_index, edge_attr,hour, week):
        z = self.concat_time(hour, week) # concatenate hour and week, (batch_size , #nodes*node_dim2 +emd )     
        z = self.dropout_layer(z)
        z = self.fc(z)    # (batch_size , node_dim2 )
        return z

    def decode(self, z, edge_index, hour,week):
        con_emb = self.concat_time(hour, week) # concatenate hour and week
        z = torch.cat((z, con_emb), dim=-1)
        z = self.fc2(z)     # (batch_size_embd , #nodes*node_dim2 )
        z = F.relu(z)
        z = z.view(-1, self.node_dim2)    #(#nodes*batch_size, node_dim2)
        edge_attr_pred = self.decoder(z, edge_index)
        if self.sigmoid:
            edge_attr_pred = torch.sigmoid(edge_attr_pred)
        return edge_attr_pred
    
    def forward(self, x, edge_index, edge_attr, hour, week):
        z = self.encode(x, edge_index, edge_attr, hour, week)
        adj = self.decode(z, edge_index,hour, week)
        return adj   

    def concat_time(self, hour, week):
        # concatenate hour and week
        emb_h = self.hour_embedding(hour)
        emb_w = self.week_embedding(week)
        z = torch.cat((emb_h, emb_w), dim=-1)
        return z 
