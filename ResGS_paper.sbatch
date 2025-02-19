#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=ResGS_paper                  # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_ResGS_paper.log        # Log-file (important!)
#SBATCH --cpus-per-task=4                       # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G                        # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                            # Number of GPU's
#SBATCH --partition=2080ti
#SBATCH --time=6-00:00:00                       # Max execution time
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
python -u ResGS_paper.py -m ResGS_paper --data SNP_markers -w "ResGS_paper with default Ridge model" -p ep_res,de_res,FESSEp_res,FESSEa_res,size_res,MUSC_res -f basic_train_function