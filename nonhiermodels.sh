#!/bin/bash
#SBATCH -t 40:00:00
#SBATCH --partition=gpu_shared_course

module load pre2019
module load Miniconda3/4.3.27
source activate prototype2

python train.py --dir "nonhier30boystotal" 

