#!/bin/bash
#SBATCH --job-name=MIR_train
#SBATCH --time=15:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --output=train_log.%j
#SBATCH --account=PAS2089

python train.py

