# Script to initialize the environnement used for the project

# Set the name of the environment
envname=TFE

# Create the environment
# from https://github.com/montefiore-institute/alan-cluster/blob/master/README.md
conda create -n $envname python=3.9.20 -c conda-forge
conda activate $envname
conda install pip
conda install -c conda-forge cupy

# All packages need to be installed via the requirements file and pip