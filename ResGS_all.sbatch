#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=ResGS_all                    # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_ResGS_all.log          # Log-file (important!)
#SBATCH --cpus-per-task=4                       # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G                        # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:2                            # Number of GPU's
#SBATCH --time=7-00:00:00                       # Max execution time
#SBATCH --partition=2080ti
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
python -u train.py --all -m ResGS_paper_very_big_expanded -d SNP_markers -p ep_res,de_res,FESSEp_res,FESSEa_res -w "Test ResGs_all" -f train_AdamW_100