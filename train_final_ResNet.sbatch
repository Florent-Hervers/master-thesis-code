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

python -u train.py -m Test_ResNet_ep_res -d SNP_markers -p ep_res -w "Final ResNet for ep_res" -f train_AdamW_LinearLR -o "Trained_models/ResNet_ep_res"
python -u train.py -m Test_ResNet_de_res -d SNP_markers -p de_res -w "Final ResNet for de_res" -f train_AdamW -o "Trained_models/ResNet_de_res"
python -u train.py -m Test_ResNet_FESSEp_res -d SNP_markers -p FESSEp_res -w "Final ResNet for FESSEp_res" -f train_AdamW -o "Trained_models/ResNet_FESSEp_res"
python -u train.py -m Test_ResNet_FESSEa_res -d SNP_markers -p FESSEa_res -w "Final ResNet for FESSEa_res" -f train_AdamW_LinearLR_40 -o "Trained_models/ResNet_FESSEa_res"
python -u train.py -m Test_ResNet_size_res -d SNP_markers -p size_res -w "Final ResNet for size_res" -f train_AdamW -o "Trained_models/ResNet_size_res"
python -u train.py -m Test_ResNet_MUSC_res -d SNP_markers -p MUSC_res -w "Final ResNet for MUSC_res" -f train_AdamW_LinearLR -o "Trained_models/ResNet_MUSC_res"