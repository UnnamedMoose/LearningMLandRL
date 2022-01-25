#!/bin/bash

#SBATCH --ntasks-per-node=28
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=11GB
#SBATCH --job-name="Some job"

module load cuda/11.0

python3 example_04_RLbasics_DQN_stableBaselines_cartpole.py
