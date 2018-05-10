#!/usr/bin/env bash

# Source bashrc
#source $HOME/.bashrc

sbatch --account=rpp-bengioy --gres=gpu:1 --qos=unkillable --mem=4G
--time=12:00:00 --mail-type=ALL --mail-user=mattcscicluna@gmail.com run_cedar.sh

#rpp-bengioy for cedar
#def-bengioy for graham
