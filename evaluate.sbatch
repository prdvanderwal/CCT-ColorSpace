#!/bin/bash

## setting output file name
#SBATCH -J EVAL_RGB
#SBATCH --output=slurm-%j_%x.out



## by defult you can run 4 jobs in parallel on the cluster
## if you need to run more jobs you can use the following command:

## CPU core numbers
##SBATCH -c 1

## Set time limit due to maintenance
#SBATCH --time=00-00:20:00

##-------------  gpu   ------------
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --mem=6GB


set -e
set -x


##set TORCH_USE_CUDA_DSA=1

source $HOME/venvs/pushpull-vit/bin/activate


python3 evaluate.py --seed 88 --name 'EVAL_RGB_final_v2' --path '/home4/p315895/DeepLearningColor/wandb/run-20240415_213458-1415otih/files/model_best.ckpt'

echo "script finished"

<<'END'





