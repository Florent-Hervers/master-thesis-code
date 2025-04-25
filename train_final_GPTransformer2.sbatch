#!/usr/bin/env bash

# Slurm arguments
#
#SBATCH --job-name=GPTransformer2               # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_GPTransformer2.log     # Log-file (important!)
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

python -u GPTransformer.py -s tokenization -e learned -m Test_GPTransformer2_4mer -d SNP_markers -p ep_res -w "Final GPTransformer 2 with 4-mer for ep_res" -f train_AdamW_100 -t "../Data/tokenized_genotype_5_4.csv" -o "Trained_models/GPTransformer2_4mer_ep_res"
python -u GPTransformer.py -s tokenization -e learned -m Test_GPTransformer2_4mer -d SNP_markers -p de_res -w "Final GPTransformer 2 with 4-mer for de_res" -f train_AdamW_100 -t "../Data/tokenized_genotype_5_4.csv" -o "Trained_models/GPTransformer2_4mer_de_res"
python -u GPTransformer.py -s tokenization -e learned -m Test_GPTransformer2_4mer -d SNP_markers -p FESSEp_res -w "Final GPTransformer 2 with 4-mer for FESSEp_res" -f train_AdamW_100 -t "../Data/tokenized_genotype_5_4.csv" -o "Trained_models/GPTransformer2_4mer_FESSEp_res"
python -u GPTransformer.py -s tokenization -e learned -m Test_GPTransformer2_4mer -d SNP_markers -p FESSEa_res -w "Final GPTransformer 2 with 4-mer for FESSEa_res" -f train_AdamW_100 -t "../Data/tokenized_genotype_5_4.csv" -o "Trained_models/GPTransformer2_4mer_FESSEa_res"
python -u GPTransformer.py -s tokenization -e learned -m Test_GPTransformer2_4mer -d SNP_markers -p size_res -w "Final GPTransformer 2 with 4-mer for size_res" -f train_AdamW_100 -t "../Data/tokenized_genotype_5_4.csv" -o "Trained_models/GPTransformer2_4mer_size_res"
python -u GPTransformer.py -s tokenization -e learned -m Test_GPTransformer2_4mer -d SNP_markers -p MUSC_res -w "Final GPTransformer 2 with 4-mer for MUSC_res" -f train_AdamW_100 -t "../Data/tokenized_genotype_5_4.csv" -o "Trained_models/GPTransformer2_4mer_MUSC_res"

python -u GPTransformer.py -s tokenization -e learned -m Test_GPTransformer2_aug -d SNP_markers -p ep_res -w "Final GPTransformer 2 with augmented vocab for ep_res" -f train_AdamW_100 -t "../Data/tokenized_genotype_9_4.csv" -o "Trained_models/GPTransformer2_aug_ep_res"
python -u GPTransformer.py -s tokenization -e learned -m Test_GPTransformer2_aug -d SNP_markers -p de_res -w "Final GPTransformer 2 with augmented vocab for de_res" -f train_AdamW_100 -t "../Data/tokenized_genotype_9_4.csv" -o "Trained_models/GPTransformer2_aug_de_res"
python -u GPTransformer.py -s tokenization -e learned -m Test_GPTransformer2_aug -d SNP_markers -p FESSEp_res -w "Final GPTransformer 2 with augmented vocab for FESSEp_res" -f train_AdamW_100 -t "../Data/tokenized_genotype_9_4.csv" -o "Trained_models/GPTransformer2_aug_FESSEp_res"
python -u GPTransformer.py -s tokenization -e learned -m Test_GPTransformer2_aug -d SNP_markers -p FESSEa_res -w "Final GPTransformer 2 with augmented vocab for FESSEa_res" -f train_AdamW_100 -t "../Data/tokenized_genotype_9_4.csv" -o "Trained_models/GPTransformer2_aug_FESSEa_res"
python -u GPTransformer.py -s tokenization -e learned -m Test_GPTransformer2_aug -d SNP_markers -p size_res -w "Final GPTransformer 2 with augmented vocab for size_res" -f train_AdamW_100 -t "../Data/tokenized_genotype_9_4.csv" -o "Trained_models/GPTransformer2_aug_size_res"
python -u GPTransformer.py -s tokenization -e learned -m Test_GPTransformer2_aug -d SNP_markers -p MUSC_res -w "Final GPTransformer 2 with augmented vocab for MUSC_res" -f train_AdamW_100 -t "../Data/tokenized_genotype_9_4.csv" -o "Trained_models/GPTransformer2_aug_MUSC_res"


