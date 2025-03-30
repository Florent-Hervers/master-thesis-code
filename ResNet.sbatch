#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=ResNet                       # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_ResNet.log             # Log-file (important!)
#SBATCH --cpus-per-task=4                       # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G                        # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                            # Number of GPU's
#SBATCH --time=7-00:00:00                       # Max execution time
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
python -u train.py -m ResNet_10_2_block -d SNP_markers -p ep_res,de_res,FESSEp_res,FESSEa_res,size_res,MUSC_res -w "ResNet_10_2_block with larger batch size" -f train_AdamW