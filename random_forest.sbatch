#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=random_forest                # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_random_forest.log      # Log-file (important!)
#SBATCH --cpus-per-task=32                      # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=125M                      # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                            # Number of GPU's
#SBATCH --time=4-00:00:00                       # Max execution time
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
python -u random_forest_all.py
