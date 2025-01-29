#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=Shallow_MLP                  # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_Shallow_MLP.log        # Log-file (important!)
#SBATCH --cpus-per-task=4                       # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=2G                        # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                            # Number of GPU's
#SBATCH --time=2-00:00:00                       # Max execution time
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/

for i in $(seq 2 5)
    do  
    python -u train.py -t Shallow_MLP_promising -d SNP_markers -p ep_res,de_res,FESSEp_res,FESSEa_res,size_res,MUSC_res -w "Test promising hyperparameters $i"
    done