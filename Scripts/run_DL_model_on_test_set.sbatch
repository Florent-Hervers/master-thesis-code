#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=test_DL                      # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_test_DL.log            # Log-file (important!)
#SBATCH --cpus-per-task=4                       # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G                        # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:2                            # Number of GPU's
#SBATCH --time=7-00:00:00                       # Max execution time
#SBATCH --partition=2080ti
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/
python -u run_DL_model_on_test_set.py