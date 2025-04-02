#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=GPTransformer                # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_GPTransformer.log      # Log-file (important!)
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
python -u GPTransformer.py -s tokenization -e learned -m GPTransformer2 -d SNP_markers -p ep_res -w "GPTransformer 2 with Gelu" -f train_AdamW_100 -t "../Data/tokenized_genotype_5_8.csv"
