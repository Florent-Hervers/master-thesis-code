#!/usr/bin/env bash

# Slurm arguments
#
#SBATCH --job-name=dataset_size                 # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_dataset_size.log       # Log-file (important!)
#SBATCH --cpus-per-task=4                       # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=8G                        # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                            # Number of GPU's
#SBATCH --time=10-00:00:00                       # Max execution time
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/

model="ShallowMLP"

for phenotype in "ep_res" # "de_res" "FESSEp_res" "FESSEa_res" "size_res" "MUSC_res"
do 
    for dataset in "200" "500" "1k" "2k" "5k" "10k"
    do
        python -u train.py -m Test_${model}_$phenotype -d SNP_markers_$dataset -p $phenotype -w "Try $model with $dataset training examples on $phenotype" -f basic_train_function -o "Dataset_models/${dataset}dataset_$model_$phenotype"
    done
done