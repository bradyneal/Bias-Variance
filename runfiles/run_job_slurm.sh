#!/usr/bin/env bash

sbatch --gres=gpu -C'gpu6gb' --qos=unkillable --mail-type=ALL --mail-user=mattcscicluna@gmail.com matt_run.sh