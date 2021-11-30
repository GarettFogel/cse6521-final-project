#!/bin/bash
#SBATCH --job-name=MIR_train2
#SBATCH --time=1:00:00
#SBATCH --nodes=1 
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=2
#SBATCH --output=train_log.%j
#SBATCH --account=PAS2089

python -W ignore train.py

