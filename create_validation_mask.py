import pandas as pd
import numpy as np

from dataset import SNPmarkersDataset

if __name__ == "__main__":
    """
    Create a new file where the validation samples are also masked (for the GBLUP)
    """
    df = pd.read_csv("../Data/BBBDL_pheno_2023bbb_0twins_6traits_mask_processed.csv", index_col = 1)
    validation_dataset = SNPmarkersDataset(mode = "validation")

    for pheno in df.drop("col_1", axis = 1).columns:
        # Store the number of samples before the processing for the check done afterwards
        length_before = df[pheno].dropna().shape[0]
        
        index = validation_dataset.phenotypes[pheno].index
        for i, _ in df.iterrows():
            if i in index:
                df.loc[i, pheno] = np.nan
    
        if length_before - len(index) != df[pheno].dropna().shape[0]:
            raise Exception(f"Error for phenotype {pheno}. The number of remaining samples after the processing ({df[pheno].dropna().shape[0]}) doesn't match the expected number ({length_before - len(index)}) !")
    
    # Reset the index and reorder the columns correctly
    df.reset_index(inplace= True)
    order = df.columns.to_list()
    tmp = order[0]
    order[0] = order[1]
    order[1] = tmp
    df.to_csv("../Data/BBBDL_pheno_2023bbb_0twins_6traits_validation_mask", sep = "\t", header=False, columns=order, index=False, na_rep= "NA")