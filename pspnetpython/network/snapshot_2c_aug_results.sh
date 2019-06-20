#!/bin/bash

# Exit immediately if a command exits with a non-zero statu
set -e

python AccAndConfMatrixAnyDb_handdb.py --model_name "train_iter_100000" --snapshot_name "snapshot_1_2c_aug" --base_name "val" --run_acc "True"
python AccAndConfMatrixAnyDb_handdb.py --model_name "train_iter_100000" --snapshot_name "snapshot_2_2c_aug" --base_name "val" --run_acc "True"
python AccAndConfMatrixAnyDb_handdb.py --model_name "train_iter_100000" --snapshot_name "snapshot_3_2c_aug" --base_name "val" --run_acc "True"
python AccAndConfMatrixAnyDb_handdb.py --model_name "train_iter_100000" --snapshot_name "snapshot_4_2c_aug" --base_name "val" --run_acc "True"
python AccAndConfMatrixAnyDb_handdb.py --model_name "train_iter_100000" --snapshot_name "snapshot_5_2c_aug" --base_name "val" --run_acc "True"