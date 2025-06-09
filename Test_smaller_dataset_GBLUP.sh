#!/bin/bash

# Script used to execute the GBLUP model using less samples in the training set

# ---------------------------------------------------------------------------------------
#                               Parameters defintions
# ---------------------------------------------------------------------------------------

# Folder name to store result
folder='Test_smaller_Datasets_GBLUP'

# Prefix for the subfolder name in the folder with results
phenotypes=("ep_res")

# Filepath to the bed file
bed_file='../Data/BBBDL_BBB2023_MD'

# Filepath to the masked phenotype file
pheno_file_prefix='../Data/BBBDL_pheno_2023bbb_0twins_6traits_train'

# Number of phenotype in the pheno file 
end=1

# Define the prefix for generated files depending of the command who created them
grm_name='GRM'
reml_name='reml'
gblup_name='gblup'

# Define nb of threads to use for the computations
n_threads=8

method="VanRaden"
# ---------------------------------------------------------------------------------------
#                               START OF THE SCRIPT
# ---------------------------------------------------------------------------------------

mkdir $folder
for dataset_size in "200" "500" "1k" "2k" "5k" "10k"
do
    
    mkdir $folder/$dataset_size
    if [ "$method" == "Yang" ]; then
        power=-1
    else
        power=0
    fi
    ldak6.linux --calc-kins-direct $folder/$dataset_size/$grm_name\_$method --bfile $bed_file --ignore-weights YES --power $power --allow-multi YES --max-threads $n_threads

    for i in $(seq 0 $((end - 1)))
    do 
        path="$folder/$dataset_size/${phenotypes[$i]}/"
        mkdir $path
        ldak6.linux --reml $path$reml_name\_$method --grm $folder/$dataset_size/$grm_name\_$method --pheno "${pheno_file_prefix}_$dataset_size" --mpheno $(( i + 1)) --max-threads $n_threads
        ldak6.linux --calc-blups $path$gblup_name\_$method --remlfile "$path${reml_name}_$method.reml" --grm $folder/$dataset_size/$grm_name\_$method --bfile $bed_file --max-threads $n_threads --allow-multi YES --check-root NO
    done
done