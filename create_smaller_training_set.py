import pandas as pd
import numpy as np

from tqdm import tqdm
from dataset import SNPmarkersDataset

if __name__ == "__main__":
    """
    Create new files where all phenotypes but the ones from the smaller training set are masked 
    to be used to estimate the impact of the training size on the GBLUP
    """
    for mode in tqdm(["train_200","train_5k", "train_2k", "train_10k", "train_500", "train_1k"]):
        df = pd.read_csv("../Data/BBBDL_pheno_2023bbb_0twins_6traits_mask_processed.csv", index_col = 1)
        dataset = SNPmarkersDataset(mode = mode, skip_check= True)

        for pheno in df.drop("col_1", axis = 1).columns:
            dataset.set_phenotypes = pheno
            index = dataset.phenotypes[pheno].index
            for i, _ in df.iterrows():
                if i not in index:
                    df.loc[i, pheno] = np.nan
        
            if len(dataset) != df[pheno].dropna().shape[0]:
                raise Exception(f"Error for phenotype {pheno}. The number of remaining samples after the processing ({df[pheno].dropna().shape[0]}) doesn't match the expected number ({len(dataset)}) !")
        
        # Reset the index and reorder the columns correctly
        df.reset_index(inplace= True)
        order = df.columns.to_list()
        tmp = order[0]
        order[0] = order[1]
        order[1] = tmp
        df.to_csv(f"../Data/BBBDL_pheno_2023bbb_0twins_6traits_{mode}", sep = "\t", header=False, columns=order, index=False, na_rep= "NA")