#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=ResGS_paper                  # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_ResGS_paper.log        # Log-file (important!)
#SBATCH --cpus-per-task=4                       # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G                        # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:2                            # Number of GPU's
#SBATCH --partition=2080ti
#SBATCH --time=6-00:00:00                       # Max execution time
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
python -u train_residual.py -m ResGS_paper_very_big_expanded --data SNP_markers -w "Test very big ResGS_paper with target size = 8400" -p ep_res,de_res,FESSEp_res,FESSEa_res,size_res,MUSC_res -f basic_train_function