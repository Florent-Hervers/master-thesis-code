from bed_reader import open_bed
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class SNPmarkersDataset(Dataset):
    def __init__(self, mode = "train"):
        """Create a class following the pytorch template.

        Args:
            mode (str, optional): type of the data stored. Only 'train' or 'test' or 'validation' are accepted values. Defaults to "train".

        Raises:
            AttributeError: if the mode doesn't respect the described format.
        """
        if mode != "train" and mode != 'test' and mode != "validation":
            raise AttributeError("the mode argument must be either 'train' or 'test' or 'validation'!")
        
        self.mode = mode

        # Get phenotypes depending on the mode 

        pheno_masked_df = pd.read_csv("../Data/BBBDL_pheno_2023bbb_0twins_6traits_mask_processed.csv", index_col= 1)
        pheno_masked_df = pheno_masked_df.drop(["col_1"], axis = 1).dropna(how="all")

        if mode == "train": 
            # TODO: update the code to include proprely the pheno_5 and 6
            self.pheno = pheno_masked_df.iloc[:-1169].drop(["pheno_5", "pheno_6"], axis = 1).dropna()
        
        if mode == "validation":
            # TODO: update the code to include proprely the pheno_5 and 6
            self.pheno = pheno_masked_df.iloc[-1169:].drop(["pheno_5", "pheno_6"], axis = 1).dropna()

        if mode == "test":
            all_pheno_df = pd.read_csv("../Data/BBBDL_pheno_20000bbb_6traits_processed.csv", index_col=1)
            self.pheno = all_pheno_df.loc[(~ all_pheno_df.index.isin(pheno_masked_df.index)).tolist()]
            self.pheno = self.pheno.drop(["col_1"], axis = 1)

        # Fetch the data of the indexes selected in the self.pheno variable
        indexes = list(map(lambda a: int(a.split("_")[-1]) - 1, self.pheno.index.to_list()))
        with open_bed("../Data/BBBDL_BBB2023_MD.bed") as bed:
            self.input = bed.read(np.s_[indexes,:], num_threads=8, dtype="int8")

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.pheno.iloc[idx]