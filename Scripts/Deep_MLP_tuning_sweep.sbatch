#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=tuning_MLP                       # Name of the job 
#SBATCH --export=ALL                                # Export all environment variables
#SBATCH --output=results_tuning_Deep_MLP.log        # Log-file (important!)
#SBATCH --cpus-per-task=4                           # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=4G                            # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                                # Number of GPU's
#SBATCH --time=10-00:00:00                          # Max execution time
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/ 
python -u sweep.py -s Deep_MLP_tuning -m Deep_MLP -d SNP_markers -p FESSEp_res,FESSEa_res,MUSC_res -f basic_train_function