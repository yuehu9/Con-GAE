
# train
CUDA_VISIBLE_DEVICES=0 python3 train.py --model 'ConGAE' --verbal False & 
P1=$! 


CUDA_VISIBLE_DEVICES=0 python3 train.py  --model 'ConGAE_t' & 
P2=$! 


CUDA_VISIBLE_DEVICES=0 python3 train.py --model 'ConGAE_sp' & 
P3=$! 

wait $P0 $P1 $P2 

# test

python3 test.py --model 'ConGAE'

python3 test.py --model 'ConGAE_t'

python3 test.py --model 'ConGAE_sp'
