#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=Tokenization                 # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_Tokenization.log       # Log-file (important!)
#SBATCH --cpus-per-task=4                       # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=16G                       # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:0                            # Number of GPU's
#SBATCH --time=1-00:00:00                       # Max execution time
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
python -u tokenize_sequences.py -v 3 -t 4