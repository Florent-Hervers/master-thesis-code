#!/usr/bin/env bash

# Slurm arguments
#
#SBATCH --job-name=LCLNN                        # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_LCLNN.log              # Log-file (important!)
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

python -u train.py -m Test_LCLNN_ep_res -d SNP_markers -p ep_res -w "Final LCLNN for ep_res" -f basic_train_function -o "Trained_models/LCLNN_ep_res"
python -u train.py -m Test_LCLNN_de_res -d SNP_markers -p de_res -w "Final LCLNN for de_res" -f basic_train_function -o "Trained_models/LCLNN_de_res"
python -u train.py -m Test_LCLNN_FESSEp_res -d SNP_markers -p FESSEp_res -w "Final LCLNN for FESSEp_res" -f basic_train_function -o "Trained_models/LCLNN_FESSEp_res"
python -u train.py -m Test_LCLNN_FESSEa_res -d SNP_markers -p FESSEa_res -w "Final LCLNN for FESSEa_res" -f basic_train_function -o "Trained_models/LCLNN_FESSEa_res"
python -u train.py -m Test_LCLNN_size_res -d SNP_markers -p size_res -w "Final LCLNN for size_res" -f basic_train_function -o "Trained_models/LCLNN_size_res"
python -u train.py -m Test_LCLNN_MUSC_res -d SNP_markers -p MUSC_res -w "Final LCLNN for MUSC_res" -f basic_train_function -o "Trained_models/LCLNN_MUSC_res"