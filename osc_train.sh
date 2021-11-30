#!/bin/bash
#SBATCH --job-name=MIR_train
#SBATCH --time=20:00
#SBATCH --nodes=1 
#SBATCH --gpus-per-node=2
#SBATCH --output=%x.%A.log
#SBATCH --gpus-per-node=1
#SBATCH --output=train_log.%j
#SBATCH --account=PAS2089

python train.py


