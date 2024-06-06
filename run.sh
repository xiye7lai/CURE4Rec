#!/bin/bash

# Define the parameters
datasets=("gowalla")
models=("mf" "bpr")
groups=(10)
learns=("receraser" "ultraue")
deltypes=("random" "core" "edge")
delpers=(5)
verbose=2

# Loop through all combinations of parameters
for dataset in "${datasets[@]}"; do
    for model in "${models[@]}"; do
        for group in "${groups[@]}"; do
            for learn in "${learns[@]}"; do
                for deltype in "${deltypes[@]}"; do
                  for delper in "${delpers[@]}"; do
                    # Construct the command
                      log="./log/${delper}_${dataset}_${model}_${learn}_${deltype}_${group}.txt"
                      cmd="nohup python main.py --dataset $dataset --model $model --group $group --learn $learn --deltype $deltype --delper $delper --verbose $verbose > $log 2>&1 &"

                      # Print and execute the command
                      echo "Running: sh"
                      eval $cmd
                  done
                done
            done
        done
    done
done
