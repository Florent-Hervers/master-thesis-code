#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=tuning_ResGS_paper               # Name of the job 
#SBATCH --export=ALL                                # Export all environment variables
#SBATCH --output=results_tuning_ResGS_paper.log     # Log-file (important!)
#SBATCH --cpus-per-task=4                           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G                            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:2                                # Number of GPU's
#SBATCH --partition=2080ti
#SBATCH --time=10-00:00:00                          # Max execution time                          


# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
python -u sweep.py --residual -s "ResGS_paper tuning" -m ResGS_paper_very_big_expanded -d SNP_markers -p size_res,MUSC_res -f basic_train_function