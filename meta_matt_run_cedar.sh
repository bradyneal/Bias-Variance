#!/usr/bin/env bash

# Source bashrc
#source $HOME/.bashrc

sbatch --account=rpp-bengioy --gres=gpu:1 --qos=high --mem=4G --time=34:00:00 --mail-type=ALL
--mail-user=mattcscicluna@gmail.com matt_run_cedar.sh

#rpp-bengioy for cedar
#def-bengioy for graham
