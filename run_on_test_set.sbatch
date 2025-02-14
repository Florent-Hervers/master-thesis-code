#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=test                             # Name of the job 
#SBATCH --export=ALL                                # Export all environment variables
#SBATCH --output=results_test_set.log               # Log-file (important!)
#SBATCH --cpus-per-task=32                          # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=1G                            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                                # Number of GPU's
#SBATCH --time=2-00:00:00                           # Max execution time                          


# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
python -u run_on_test_set.py -m Ridge
python -u run_on_test_set.py -m Random_forest
python -u run_on_test_set.py -m SVM