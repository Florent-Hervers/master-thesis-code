#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=test_Deep_MLP                    # Name of the job 
#SBATCH --export=ALL                                # Export all environment variables
#SBATCH --output=test_Deep_MLP.log                  # Log-file (important!)
#SBATCH --cpus-per-task=4                           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=1G                            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                                # Number of GPU's
#SBATCH --time=1-00:00:00                           # Max execution time
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
python -u train.py -t test -d SNP_markers -p ep_res -m Deep_MLP -f basic_train_function