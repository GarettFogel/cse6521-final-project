#!/bin/bash
#SBATCH --job-name=MIR_train
#SBATCH --time=2:00
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --output=%x.%A.log
#SBATCH --signal=B:USR1@60
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=myjob.out.%j
#SBATCH --account=PAS2089

python test.py


