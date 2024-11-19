#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=random_forest                        # Name of the job 
#SBATCH --export=ALL                                    # Export all environment variables
#SBATCH --output=results_random_forest_2_1000.log         # Log-file (important!)
#SBATCH --cpus-per-task=64                              # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=100M                              # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                                    # Number of GPU's
#SBATCH --time=6-00:00:00                               # Max execution time
#

# Note that the same environement was used by all random_forests

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
python -u random_forest.py
