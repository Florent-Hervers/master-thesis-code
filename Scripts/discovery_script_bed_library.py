import numpy as np
from bed_reader import open_bed

with open_bed("../Data/BBBDL_BBB2023_MD.bed") as bed:
    # Read and find the size of the SNP arrays
    val = bed.read()
    print(val.shape)

    # Display available informations about the data
    print(bed.properties)
    print(np.unique(bed.chromosome))
    