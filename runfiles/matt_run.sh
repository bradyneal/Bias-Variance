#!/usr/bin/env bash

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export PYTHONUNBUFFERED=1
source activate inf-path
echo Running on $HOSTNAME

python matt_test.py