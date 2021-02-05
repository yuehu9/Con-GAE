### Note on Data ####
The appended Uber movement data is for review process only.


## Requirements
This code was written using python 3.6, Pytorch 1.4.0, Pytorch Geometric 1.4.3
Oterh required libraries include: numpy, scipy, pandas etc.

## Usage
First chage directory to /src

Before evaluation, generate the synthetic test data by:
```
sh ../data/simulate_test_data_sp.sh
sh ../data/simulate_test_data_st.sh
```

To evaulate trained model, run:
```
python3 test.py --model 'ConGAE'
python3 test.py --model 'ConGAE_t'
python3 test.py --model 'ConGAE_sp'

```
for main model ConGAE and variants ConGAE_t, ConGAE_sp repectively.

To train model with configurations stated in paper, run:
```
python3 train.py --model 'ConGAE'
python3 train.py  --model 'ConGAE_t' 
python3 train.py --model 'ConGAE_sp' 
```
for main model ConGAE and variants ConGAE_t, ConGAE_sp repectively.

Full command for ConGAE training is:
```
python3 train.py --model 'ConGAE' --randomseed 1 '--train_epoch 150 --lr 5e-5 --dropout_p 0.2 --adj_drop 0.2 --verbal False --input_dim 4 --node_dim1 300 --node_dim2 150 --encode_dim 150 --hour_emb --week_emb'--decoder 'concatDec'
```

ConGAE model has two graph convulutiona layers. For deep model with more graph convulutiona layers, use the deepConGAE variant:
```
python3 train.py --model 'deepConGAE' --hidden_list 300 150 150 --encode_dim 150 --decode_dim 150

```
where hidden_list contains the node embedding dimension of each layer of GCN

The model variant ConGAE-fc uses fully connected layers for spacial info, thus use dataframe instead of graph format for input. It is trained and tested separately. To evaluate model, run:
```
python3 ConGAE-fc-train-test.py --test True
```
or set test as False to train the model anew.