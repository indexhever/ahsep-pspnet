#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

python AccAndConfMatrixAnyDb_handdb.py --model_name "train_iter_100000" --snapshot_name "snapshot_1_2c_4000_aug_4" --base_name "val" --run_acc "True"
python AccAndConfMatrixAnyDb_handdb.py --model_name "train_iter_100000" --snapshot_name "snapshot_1_2c_4000_aug_5" --base_name "val" --run_acc "True"