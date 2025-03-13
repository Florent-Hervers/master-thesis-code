#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=SVM                          # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_SVM_15.log             # Log-file (important!)
#SBATCH --cpus-per-task=64                      # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=250M                      # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                            # Number of GPU's
#SBATCH --time=10-00:00:00                      # Max execution time
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
python -u SVM.py