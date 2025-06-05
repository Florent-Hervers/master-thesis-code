#!/usr/bin/env bash

# Slurm arguments
#
#SBATCH --job-name=Shallow_MLP                  # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_Shallow_MLP.log        # Log-file (important!)
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

python -u train.py -m Test_ShallowMLP_ep_res -d SNP_markers -p ep_res -w "Final Shallow_MLP for ep_res" -f basic_train_function -o "Trained_models/ShallowMLP_ep_res"
python -u train.py -m Test_ShallowMLP_de_res -d SNP_markers -p de_res -w "Final Shallow_MLP for de_res" -f basic_train_function -o "Trained_models/ShallowMLP_de_res"
python -u train.py -m Test_ShallowMLP_FESSEp_res -d SNP_markers -p FESSEp_res -w "Final Shallow_MLP for FESSEp_res" -f basic_train_function -o "Trained_models/ShallowMLP_FESSEp_res"
python -u train.py -m Test_ShallowMLP_FESSEa_res -d SNP_markers -p FESSEa_res -w "Final Shallow_MLP for FESSEa_res" -f basic_train_function -o "Trained_models/ShallowMLP_FESSEa_res"
python -u train.py -m Test_ShallowMLP_size_res -d SNP_markers -p size_res -w "Final Shallow_MLP for size_res" -f basic_train_function -o "Trained_models/ShallowMLP_size_res"
python -u train.py -m Test_ShallowMLP_MUSC_res -d SNP_markers -p MUSC_res -w "Final Shallow_MLP for MUSC_res" -f basic_train_function -o "Trained_models/ShallowMLP_MUSC_res"

python -u train.py -m Test_DeepMLP_ep_res -d SNP_markers -p ep_res -w "Final Deep_MLP for ep_res" -f basic_train_function -o "Trained_models/DeepMLP_ep_res"
python -u train.py -m Test_DeepMLP_de_res -d SNP_markers -p de_res -w "Final Deep_MLP for de_res" -f basic_train_function -o "Trained_models/DeepMLP_de_res"
python -u train.py -m Test_DeepMLP_FESSEp_res -d SNP_markers -p FESSEp_res -w "Final Deep_MLP for FESSEp_res" -f basic_train_function -o "Trained_models/DeepMLP_FESSEp_res"
python -u train.py -m Test_DeepMLP_FESSEa_res -d SNP_markers -p FESSEa_res -w "Final Deep_MLP for FESSEa_res" -f basic_train_function -o "Trained_models/DeepMLP_FESSEa_res"
python -u train.py -m Test_DeepMLP_size_res -d SNP_markers -p size_res -w "Final Deep_MLP for size_res" -f basic_train_function -o "Trained_models/DeepMLP_size_res"
python -u train.py -m Test_DeepMLP_MUSC_res -d SNP_markers -p MUSC_res -w "Final Deep_MLP for MUSC_res" -f basic_train_function -o "Trained_models/DeepMLP_MUSC_res"