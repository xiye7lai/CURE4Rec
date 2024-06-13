#!/bin/bash

# Define the parameters
dataset=("ml-100k")
model=("wmf")
group=(10)
learn=("retrain")
deltype=("random")
delper=(5)
verbose=2
# Construct the command
log="./log/${delper}_${dataset}_${model}_${learn}_${deltype}_${group}.txt"
cmd="nohup python main.py --dataset $dataset --model $model --group $group --learn $learn --deltype $deltype --delper $delper --verbose $verbose > $log 2>&1 &"

# run lightGCN
# cmd="nohup python lightgcn.py --dataset $dataset --group $group --learn $learn --deltype $deltype --delper $delper --verbose $verbose > $log 2>&1 &"

# Print and execute the command
echo "Running: sh"
eval $cmd