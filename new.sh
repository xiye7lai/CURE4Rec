#!/bin/bash

# Define the parameters
datasets=("adm")
learns=("sisa" "receraser" "ultrare")
deltypes=("random" "core" "edge")

# Loop through all combinations of parameters
for dataset in "${datasets[@]}"; do
  for learn in "${learns[@]}"; do
    for deltype in "${deltypes[@]}"; do
      # Construct the command
                      log="./light/${dataset}_${learn}_${deltype}.txt"
                      cmd="nohup python lightgcn.py --dataset $dataset --learn $learn --deltype $deltype > $log 2>&1 &"

                      # Print and execute the command
                      echo "Running: sh"
                      eval $cmd
        done
    done
done
