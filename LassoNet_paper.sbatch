#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=LassoNet                     # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_LassoNet_4.log         # Log-file (important!)
#SBATCH --cpus-per-task=2                       # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G                       # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                            # Number of GPU's
#SBATCH --time=7-00:00:00                       # Max execution time
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/Regression/LassoNet
python -u entry.py