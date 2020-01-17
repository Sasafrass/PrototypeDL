#!/bin/bash
#SBATCH -t -02:30:00
#SBATCH --partition=gpu_shared
#SBATCH -n 2
module load pre2019
module load Miniconda3/4.3.27
source activate prototype2

python3 train.py
