#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=test                             # Name of the job 
#SBATCH --export=ALL                                # Export all environment variables
#SBATCH --output=results_test_set_3.log             # Log-file (important!)
#SBATCH --cpus-per-task=32                          # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=1G                            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                                # Number of GPU's
#SBATCH --time=7-00:00:00                           # Max execution time                          


# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
# python -u run_on_test_set.py -m Random_forest_all
# python -u run_on_test_set.py -m XGBoost_all
# python -u run_on_test_set.py -m XGBoost
python -u run_on_test_set.py -m SVM