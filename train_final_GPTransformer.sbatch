#!/usr/bin/env bash

# Slurm arguments
#
#SBATCH --job-name=GPTransformer                # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_GPTransformer.log      # Log-file (important!)
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

python -u GPTransformer.py -s mutual_information -e frequency -m Test_GPTransformer_ep_res -d SNP_markers -p ep_res -w "Final GPTransformer model for ep_res" -f train_MSE -o Trained_models/GPTransformer_ep_res
python -u GPTransformer.py -s mutual_information -e categorical -m Test_GPTransformer_de_res -d SNP_markers -p de_res -w "Final GPTransformer model for de_res" -f train_MSE -o Trained_models/GPTransformer_de_res
python -u GPTransformer.py -s mutual_information -e categorical -m Test_GPTransformer_FESSEp_res -d SNP_markers -p FESSEp_res -w "Final GPTransformer model for FESSEp_res" -f basic_train_function -o Trained_models/GPTransformer_FESSEp_res
python -u GPTransformer.py -s mutual_information -e categorical -m Test_GPTransformer_FESSEa_res -d SNP_markers -p FESSEa_res -w "Final GPTransformer model for FESSEa_res" -f basic_train_function -o Trained_models/GPTransformer_FESSEa_res
python -u GPTransformer.py -s tokenization -e learned -m Test_GPTransformer_size_res -d SNP_markers -p size_res -w "Final GPTransformer model for size_res" -f train_AdamW_100 -o Trained_models/GPTransformer_size_res -t ../Data/tokenized_genotype_5_8.csv
python -u GPTransformer.py -s mutual_information -e categorical -m Test_GPTransformer_MUSC_res -d SNP_markers -p MUSC_res -w "Final GPTransformer model for MUSC_res" -f train_MSE -o Trained_models/GPTransformer_MUSC_res