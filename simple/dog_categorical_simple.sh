#!/bin/bash

#SBATCH --job-name="simple_dog_cat"

#SBATCH --workdir=/home/nct01/nct01068/simple

#SBATCH --output=dog_%j.out

#SBATCH --error=dog%j.err

#SBATCH --ntasks=1

#SBATCH --gres gpu:1

#SBATCH --time=48:00:00

module purge; module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python identifying_dog_breeds_simple.py
