#!/usr/bin/env bash

# Source bashrc
#source $HOME/.bashrc

sbatch --account=rpp-bengioy --gres=gpu:1 --qos=high --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 0 --num_seeds 3  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1
sbatch --account=rpp-bengioy --gres=gpu:3 --qos=high --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 3 --num_seeds 6  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1
sbatch --account=rpp-bengioy --gres=gpu:3 --qos=unkillable --mem=8G --time=5:00:00 cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 6 --num_seeds 9  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1
sbatch --account=rpp-bengioy --gres=gpu:3 --qos=low --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 9 --num_seeds 12  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1
sbatch --account=rpp-bengioy --gres=gpu:3 --qos=low --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 12 --num_seeds 15  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1
sbatch --account=rpp-bengioy --gres=gpu:3 --qos=low --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 15 --num_seeds 18  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1
sbatch --account=rpp-bengioy --gres=gpu:3 --qos=low --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 18 --num_seeds 21  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1
sbatch --account=rpp-bengioy --gres=gpu:3 --qos=low --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 21 --num_seeds 24  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1
sbatch --account=rpp-bengioy --gres=gpu:3 --qos=low --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 24 --num_seeds 27  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1
sbatch --account=rpp-bengioy --gres=gpu:3 --qos=low --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 27 --num_seeds 30  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1

sbatch --account=rpp-bengioy --gres=gpu:1 --qos=low --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 30 --num_seeds 33  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1
sbatch --account=rpp-bengioy --gres=gpu:3 --qos=low --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 33 --num_seeds 36  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1
sbatch --account=rpp-bengioy --gres=gpu:3 --qos=low --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 36 --num_seeds 39  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1
sbatch --account=rpp-bengioy --gres=gpu:3 --qos=low --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 39 --num_seeds 42  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1
sbatch --account=rpp-bengioy --gres=gpu:3 --qos=low --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 42 --num_seeds 45  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1
sbatch --account=rpp-bengioy --gres=gpu:3 --qos=low --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 45 --num_seeds 48  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com
sleep 1
sbatch --account=rpp-bengioy --gres=gpu:3 --qos=low --mem=8G cedar_run.sh
--hidden_arr 5 25 100 500 2500 10000 40000 --start_seed 48 --num_seeds 50  --learning_rate 0.01 --batch_size 100 --max_epochs 500 --save_best_model --mail-type=ALL --mail-user=mattcscicluna@gmail.com


$@


