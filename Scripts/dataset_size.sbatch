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

for model in "ResNet" "ShallowMLP"
do
    for phenotype in "ep_res" 
    do 
        for dataset in "200" "500" "1k" "2k" "5k" "10k"    
        do
            if [ "$model" = "ShallowMLP" ]; then
                train_function="basic_train_function"
            elif [ "$model" = "ResNet" ]; then
                train_function="train_AdamW_LinearLR"
            else
                echo "Erreur : modèle inconnu '$model'."
                exit 1
            fi
            python -u train.py -m Test_${model}_$phenotype -d SNP_markers_$dataset -p $phenotype -w "Try $model with $dataset training examples on $phenotype" -f $train_function -o "Dataset_models/${dataset}dataset_${model}_$phenotype"
        done
    done
done