#!/bin/bash

## setting output file name
#SBATCH -J RGB_LN
#SBATCH --output=slurm-%j_%x.out



## by defult you can run 4 jobs in parallel on the cluster
## if you need to run more jobs you can use the following command:

## CPU core numbers
##SBATCH -c 1

## Set time limit due to maintenance
#SBATCH --time=00-04:00:00

##-------------  gpu   ------------
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1
#SBATCH --mem=6GB


set -e
set -x


##set TORCH_USE_CUDA_DSA=1

source $HOME/venvs/pushpull-vit/bin/activate


python3 main.py --seed 88 --name RGB_LN_final_s88

echo "script finished"

<<'END'






