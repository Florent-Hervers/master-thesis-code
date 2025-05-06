#!/usr/bin/env bash

# Slurm arguments
#
#SBATCH --job-name=Shallow_MLP_all              # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_Shallow_MLP.log        # Log-file (important!)
#SBATCH --cpus-per-task=4                       # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G                        # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                            # Number of GPU's
#SBATCH --time=7-00:00:00                       # Max execution time
#SBATCH --partition=2080ti
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/

python -u train.py --all -m Test_ShallowMLP_all -d SNP_markers -p ep_res,de_res,FESSEp_res,FESSEa_res -w "Final Shallow_MLP_all" -f basic_train_function -o "Trained_models/ShallowMLP_all"
