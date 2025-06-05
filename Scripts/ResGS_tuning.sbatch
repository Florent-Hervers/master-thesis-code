#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=tuning_ResGS                     # Name of the job 
#SBATCH --export=ALL                                # Export all environment variables
#SBATCH --output=results_tuning_ResGS.log           # Log-file (important!)
#SBATCH --cpus-per-task=4                           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G                            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:2                                # Number of GPU's
#SBATCH --time=10-00:00:00                          # Max execution time                          


# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
python -u sweep.py -s ResGS_tuning -m ResGS_paper_very_big_expanded -d SNP_markers -p ep_res -f train_AdamW