#!/usr/bin/env bash

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source activate inf-path

# Learning rates based on best
# python3 ngc_run.py --hidden_arr 1 2 5 25 100 500 2500 10000 40000 --num_seeds 50 \
#     --learning_rate 0.005547119471592124 0.006921859910208608 0.0013130280280658597 0.01032326035197657 0.01032326035197657 0.037665768415992965 0.01032326035197657 0.005301709347576826 0.0013130280280658597 \
#     --num_train_after_split 100 --batch_size 10 --max_epochs 1000 --print_errors train_and_val --save_best_model

# Based on last

# python3 ngc_run.py --hidden_arr 500 --num_seeds 50 \
#     --learning_rate 0.01032326035197657 \
#     --num_train_after_split 100 --batch_size 10 --max_epochs 1000 --print_errors train_and_val --save_best_model  # 176008

python3 ngc_run.py --start_seed 10 --end_seed 20 \
    --learning_rate 0.039810717055349734 0.03357669821712078 0.028275708713837965 0.044864551518898836 0.021982015841173333 0.013967249047300238 \
   --batch_size 100 --max_epochs 1000 --print_errors all --save_best_model --num_layers 2 6 7 8 9 10
   
   # python3 ngc_run.py --hidden_arr 1 2 5 25 100 500 2500 10000 40000 --num_seeds 50 --start_seed 25 \
#     --learning_rate 0.005547119471592124 0.006921859910208608 0.0013130280280658597 0.01032326035197657 0.01032326035197657 0.01032326035197657 0.01032326035197657 0.0013130280280658597 0.0013130280280658597 \
#     --num_train_after_split 100 --batch_size 10 --max_epochs 1000 --print_errors train_and_val --save_best_model  # 176011

# python3 ngc_run.py --hidden_arr 1 2 5 25 100 500 2500 10000 40000 --num_seeds 50 --start_seed 25 \
#     --learning_rate 0.01 \
#     --num_train_after_split 100 --batch_size 100 --max_epochs 1000 --print_errors train_and_val --save_best_model  # 179418

# python3 ngc_run.py --hidden_arr 1 2 5 25 100 500 2500 10000 40000 --num_seeds 50 --start_seed 25 \
#     --learning_rate 0.01 \
#     --num_train_after_split 100 --batch_size 100 --max_epochs 1000 --print_errors train_and_val --save_best_model  # 179424

# python3 ngc_run.py --hidden_arr 1 2 5 25 100 500 2500 10000 40000 --num_seeds 50 --start_seed 25 \
#     --learning_rate 0.01136425106100869 0.004048237066415427 0.025659457643601627 0.008751678448220987 0.060520129619266914 0.060291575969776565 0.010903640440605806 0.0057797872150160195 0.0014770848187591037 \
#     --num_train_after_split 100 --batch_size 100 --max_epochs 1000 --print_errors train_and_val --save_best_model  # 180422

  # python3 ngc_run.py --hidden_arr 1 2 5 25 100 500 2500 10000 40000 --num_seeds 50 \
  #     --learning_rate 0.01136425106100869 0.004048237066415427 0.025659457643601627 0.008751678448220987 0.060520129619266914 0.060291575969776565 0.010903640440605806 0.0057797872150160195 0.0014770848187591037 \
  #     --num_train_after_split 100 --batch_size 100 --max_epochs 1000 --print_errors train_and_val --save_best_model  # 180424
