#!/bin/bash
#SBATCH --account=horizon_store
#SBATCH --job-name=semantic_similarity_computation_vit_L14
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=semantic_similarity_computation_%j.out
#SBATCH --error=semantic_similarity_computation_%j.err

python ./dataset_distance/cal_sim.py