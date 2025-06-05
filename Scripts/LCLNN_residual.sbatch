#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=LCLNN                        # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_LCLNN.log              # Log-file (important!)
#SBATCH --cpus-per-task=4                       # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G                        # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                            # Number of GPU's
#SBATCH --time=2-00:00:00                       # Max execution time
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
python -u train_residual.py -m LCLNN -d SNP_markers -f basic_train_function -p ep_res,de_res,FESSEp_res,FESSEa_res,size_res,MUSC_res -w "Original LCLNN trained on residuals with lr = 1e-4"