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
#SBATCH --time=6-00:00:00                       # Max execution time
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
python -u ResGS_paper.py -m ResGS_paper_big --data SNP_markers -w "Big ResGS_paper with optimal Ridge model" -p FESSEa_res,size_res,MUSC_res -f basic_train_function