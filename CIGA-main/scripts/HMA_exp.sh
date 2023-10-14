# Graph-SST5
python main.py --no_tqdm --lgma 1 --num_heads 4 --label_hidden 32 --local_mem_size 512 --global_memory_number 16 --r 0.5 --num_layers 3  --batch_size 32 --emb_dim 128 --model 'gcn' -c_dim 128 --dataset 'Graph-SST5' --seed '[1,2,3,4,5]' --contrast 0 --spu_coe 0 -c_in 'raw'  -c_rep 'feat'

# Graph-Twitter
python main.py  --no_tqdm --erm --lgma 1 --num_heads 4 --label_hidden 32 --local_mem_size 192 --global_memory_number 48 --num_layers 3  --batch_size 32 --emb_dim 128 --model 'gcn' -c_dim 128 --dataset 'Graph-Twitter' --seed '[1,2,3,4,5]' --contrast 8  --spu_coe 0 -c_in 'feat'  -c_rep 'feat'

# proteins
python main.py --no_tqdm --erm  --lgma 1  --num_heads 4 --label_hidden 16 --local_mem_size 128 --global_memory_number 32  -c_in 'raw' -c_rep 'rep'  --seed '[1,2,3,4,5,6,7,8,9,10]'  --num_layers 3 --dataset 'proteins' --contrast 0.0 --spu_coe 0 --model 'gin' --pooling 'max' --dropout 0.3 --eval_metric 'mat' 
