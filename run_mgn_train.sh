#!/bin/bash
#SBATCH --partition=gpu_irmb
#SBATCH --nodes=1
#SBATCH --time=25:00:00
#SBATCH --job-name=linearhydrogel_mgn_train
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere


singularity exec --nv /home/y0113799/container/dev.sif python -u run_model.py
