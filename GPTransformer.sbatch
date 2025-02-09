#!/usr/bin/env bash
#
# Slurm arguments
#
#SBATCH --job-name=GPTransformer                # Name of the job 
#SBATCH --export=ALL                            # Export all environment variables
#SBATCH --output=results_GPTransformer.log      # Log-file (important!)
#SBATCH --cpus-per-task=8                       # Number of CPU cores to allocate
#SBATCH --mem-per-cpu=1G                        # Memory to allocate per allocated CPU core
#SBATCH --gres=gpu:1                            # Number of GPU's
#SBATCH --time=3-00:00:00                       # Max execution time
#

# Activate your Anaconda environment
conda activate TFE

# Run your Python script
cd ~/TFE/Code/
python -u GPTransformer.py -e learned -m GPTransformer -p ep_res,de_res,FESSEp_res,FESSEa_res,size_res,MUSC_res -w "Expand output mlp with learned embedding" -f basic_train_function
python -u GPTransformer.py -e frequency -m GPTransformer -p ep_res,de_res,FESSEp_res,FESSEa_res,size_res,MUSC_res -w "Expand output mlp with frequency embedding" -f basic_train_function
python -u GPTransformer.py -e one_hot -m GPTransformer_one_hot -p ep_res,de_res,FESSEp_res,FESSEa_res,size_res,MUSC_res -w "Expand output mlp with one_hot embedding" -f basic_train_function
python -u GPTransformer.py -e categorical -m GPTransformer -p ep_res,de_res,FESSEp_res,FESSEa_res,size_res,MUSC_res -w "Expand output mlp with categorical embedding" -f basic_train_function
