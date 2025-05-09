#!/usr/bin/env bash

# Slurm arguments
#
#SBATCH --job-name=ResGS                  # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_ResGS.log        # Log-file (important!)
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

python -u train_residual.py -m Test_ResGS_ep_res -d SNP_markers -p ep_res -w "Final ResGS for ep_res" -f basic_train_function -o "Trained_models/ResGS_ep_res"
python -u train_residual.py -m Test_ResGS_de_res -d SNP_markers -p de_res -w "Final ResGS for de_res" -f basic_train_function -o "Trained_models/ResGS_de_res"
python -u train_residual.py -m Test_ResGS_FESSEp_res -d SNP_markers -p FESSEp_res -w "Final ResGS for FESSEp_res" -f basic_train_function -o "Trained_models/ResGS_FESSEp_res"
python -u train_residual.py -m Test_ResGS_FESSEa_res -d SNP_markers -p FESSEa_res -w "Final ResGS for FESSEa_res" -f basic_train_function -o "Trained_models/ResGS_FESSEa_res"
python -u train_residual.py -m Test_ResGS_size_res -d SNP_markers -p size_res -w "Final ResGS for size_res" -f basic_train_function -o "Trained_models/ResGS_size_res"
python -u train_residual.py -m Test_ResGS_MUSC_res -d SNP_markers -p MUSC_res -w "Final ResGS for MUSC_res" -f basic_train_function -o "Trained_models/ResGS_MUSC_res"