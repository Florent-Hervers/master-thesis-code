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
python -u GPTransformer.py -e frequency -t GPTransformer -p ep_res,de_res,size_res,FESSEp_res,FESSEa_res,MUSC_res -w "GPTransformer with frenquency encoding"