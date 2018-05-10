#!/usr/bin/env bash


# Source bashrc
source $HOME/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

# Run the script
python information-paths/matt_test.py