#!/usr/bin/env bash

source ~/.bashrc
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME

source activate inf-path

python information-paths/get_var.py $@