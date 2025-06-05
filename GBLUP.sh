#!/bin/bash

# Script used to execute the GBLUP model with the ldak software

# ---------------------------------------------------------------------------------------
#                               Parameters defintions
# ---------------------------------------------------------------------------------------

# Folder name to store result
folder='GBLUP'

# Prefix for the subfolder name in the folder with results
phenotypes=("ep_res" "de_res" "FESSEp_res" "FESSEa_res" "size_res" "MUSC_res")

# Filepath to the bed file
bed_file='../Data/BBBDL_BBB2023_MD'

# Filepath to the masked phenotype file
pheno_file='../Data/BBBDL_pheno_2023bbb_0twins_6traits_validation_mask'

# Number of phenotype in the pheno file 
end=6

# Define the prefix for generated files depending of the command who created them
grm_name='GRM'
reml_name='reml'
gblup_name='gblup'

# Define nb of threads to use for the computations
n_threads=8

# ---------------------------------------------------------------------------------------
#                               START OF THE SCRIPT
# ---------------------------------------------------------------------------------------

mkdir $folder
for method in "Yang" "VanRaden"
do
    if [ "$method" == "Yang" ]; then
        power=-1
    else
        power=0
    fi
    ldak6.linux --calc-kins-direct $folder/$grm_name\_$method --bfile $bed_file --ignore-weights YES --power $power --allow-multi YES --max-threads $n_threads

    for i in $(seq 0 $((end - 1)))
    do 
        path="$folder/${phenotypes[$i]}/"
        mkdir $path
        ldak6.linux --reml $path$reml_name\_$method --grm $folder/$grm_name\_$method --pheno $pheno_file --mpheno $(( i + 1)) --max-threads $n_threads
        ldak6.linux --calc-blups $path$gblup_name\_$method --remlfile "$path${reml_name}_$method.reml" --grm $folder/$grm_name\_$method --bfile $bed_file --max-threads $n_threads --allow-multi YES --check-root NO
    done
done

# ---------------------------------------------------------------------------------------
#                               ORIGINAL INSTRUCTIONS
# ---------------------------------------------------------------------------------------

####1. we have to build one GRM using Ldak
##Instructions for LDAK to construct GRM. 
               ##https://dougspeed.com/calculate-kinships/
               ##please note --power -1 is the GRM constructed based on standaized genotype (Yang) while --power 0 is without standardisation (VanRaden)
    ##$subset is file to select marker
    ##$input_bfile a bed file of genotype
    #ldak5.2.linux --calc-kins-direct $prefix1 --bfile $input_bfile --ignore-weights YES --power -1 --allow-multi YES --max-threads 4 --extract $subset

####2. REML
####please note the trait of the validation should be mask
    ##$input_pfile
    ##$trait_num indicate the column of  trait of interesting
    #ldak5.2.linux --reml $prefix2 --grm $prefix1 --pheno $input_pfile --mpheno $trait_num --max-threads 4

####3. Blup value based on the result of REML
    #ldak5.2.linux --calc-blups $prefix3 --remlfile $prefix2.reml --grm $prefix1 --bfile $input_bfile --max-threads 4 --allow-multi YES --check-root NO

####$prefix3.pred contains the blup value of candidates
####output file with suffixes .reml is variance
####.blup is the effect of each marker