# Context augmented Graph Autoencoder (Con-GAE)
Detecting extreme traffic events via a context augmented graph autoencoder

Yue Hu, Ao Qu and Dan Work, 2021

## Overview
Context augmented Graph Autoencoder (Con-GAE) aims at detecting extreme events in traffic origin-destinatin (OD) datasets. Con-GAE leverages graph embedding and context embedding techniques to capture the spatial and temporal patterns in traffic dynamics, and adopts an autoencoder framework to detect anomalies via semi-supervised learning. Pre-processed Uber Movement data is also contained in this repository for tests.

The results are reported in "Detecting extreme traffic events via a context augmented graph autoencoder" by Y.Hu et. al. published at ACM Transactions on Intelligent Systems and Technology (TIST)


## Requirements
The current version of this code was written using python 3.8, Pytorch 1.11.0, Pytorch Geometric 2.0

Other required libraries include: numpy, scipy, pandas etc.

## Usage
First chage directory to /src

To evaulate trained model, run:
```
python3 test.py --model 'ConGAE'

```
for main model ConGAE applied on NYC dataset.

To train model with configurations stated in paper, run:
```
python3 train.py --model 'ConGAE'
```
for main model ConGAE.

Full command for ConGAE training is:
```
python3 train.py --model 'ConGAE' --randomseed 1 '--train_epoch 150 --lr 1e-3 --dropout_p 0.2 --adj_drop 0.2 --verbal False --input_dim 4 --node_dim1 150 --node_dim2 50 --encode_dim 50 --hour_emb 100 --week_emb 100 --decoder 'concatDec'
```

ConGAE model has two graph convolutional layers. For deep model with more graph convolutional layers, use the deepConGAE variant:
```
python3 train.py --model 'deepConGAE' --hidden_list 300 150 150 --encode_dim 150 --decode_dim 150

```
where hidden_list contains the node embedding dimension of each layer of GCN


