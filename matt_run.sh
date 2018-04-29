#!/usr/bin/env bash

# Source bashrc
source $HOME/.bashrc
export PYTHONUNBUFFERED=1

# Run the script
python matt_test.py $@
