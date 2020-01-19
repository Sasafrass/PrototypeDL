#!/bin/bash
#SBATCH -t 8:30:00
#SBATCH --mem=16000M
#SBATCH --partition=gpu_shared_course

module load pre2019
module load Miniconda3/4.3.27
source activate prototype2

python train.py --seed 6

