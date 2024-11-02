from bed_reader import open_bed
import pandas as pd
import numpy as np
from typing import Union, List
from torch.utils.data import Dataset

class SNPmarkersDataset(Dataset):
    """Create a class following the pytorch template to manage the data. \
    Note: SNP data should be fetched using the `get_SNP()` function.

    Attributes:
        mode (str, optional): type of the data stored. Only 'train' or 'test' or 'validation' are accepted values. Defaults to "train".
        phenotypes (dict[pd.Series]): contain the phenotypes data for the given mode. Keys are phenotypes label used in the original csv files.
        skip_check (bool): Option to skip the checking used to speed object creation when testings. Defaults to False.
    """
    
    def __init__(self, mode = "train", skip_check = False):
        """Create a class following the pytorch template.

        Args:
            mode (str, optional): type of the data stored. Only 'train' or 'test' or 'validation' are accepted values. Defaults to "train".

        Raises:
            AttributeError: if the mode doesn't respect the described format.
        """
        # Define the number of samples used for the validation set
        VALIDATION_SIZE = 1000
        self.__input = open_bed("../Data/BBBDL_BBB2023_MD.bed").read(dtype="int8", num_threads= 8)
        
        if mode != "train" and mode != 'test' and mode != "validation":
            raise AttributeError("the mode argument must be either 'train' or 'test' or 'validation'!")
        
        self.mode = mode
        self.phenotypes = {}
 
        pheno_masked_df = pd.read_csv("../Data/BBBDL_pheno_2023bbb_0twins_6traits_mask_processed.csv", index_col= 1)
        pheno_masked_df = pheno_masked_df.drop(["col_1"], axis = 1).dropna(how="all")
        
        if mode == "train":   
            for pheno in pheno_masked_df.columns:
                self.phenotypes[pheno] = pheno_masked_df[pheno].dropna().iloc[:-VALIDATION_SIZE]
            
        if mode == "validation":
            for pheno in pheno_masked_df.columns:
                self.phenotypes[pheno] = pheno_masked_df[pheno].dropna().iloc[-VALIDATION_SIZE:]

        if mode == "test":
            all_pheno_df = pd.read_csv("../Data/BBBDL_pheno_20000bbb_6traits_processed.csv", index_col=1)
            all_test_samples = all_pheno_df.loc[(~ all_pheno_df.index.isin(pheno_masked_df.index)).tolist()].drop(["col_1"], axis = 1)
            for pheno in pheno_masked_df.columns:
                self.phenotypes[pheno] = all_test_samples[pheno].dropna()

        if not skip_check:
            try:
                self.check_data()
            except Exception as e:
                raise e
    
    def get_SNP(self, pheno: Union[str, List[str]] ):
        """ Fetch the SNP data related of the given phenotype of the data concerned by the mode.

        Args:
            phenotype (Union[str, List[str]]): column name of the wanted phenotypes (ie keys of phenotypes attribute). The list enable to use several phenotypes.

        Raises:
            AttributeError: if the given phenotypes isn"t found in the original phenotype dataframe columns.
            Exception: if the requested data contain any missing values   
        
        Returns:
            List: SNP data for the given mode and phenotype
        """

        if set(pheno) <= set(self.phenotypes.keys()) or (type(pheno) == str and pheno in self.phenotypes.keys()):
            if type(pheno) == str:
                pheno = [pheno]
            
            indexes = pd.DataFrame([self.phenotypes[value] for value in pheno]).transpose().index.to_list()
            indexes = list(map(lambda a: int(a.split("_")[-1]) - 1, indexes))
            data = self.__input[indexes,:]
            
            # Check that data is free of missing values
            classes, counts = np.unique(data, return_counts=True)
            if classes.all() != np.array([0, 1, 2]).all():
                raise Exception(f"There are {counts[np.where(classes == -127)[0][0]]} missing values in the data!")

            return data
        
        else:
            raise AttributeError(f"Unknown phenotype asked. The possible values are {self.phenotypes.keys()}")
        
    def check_data(self):
        """Perform some verifications on the data (of the current object) to be sure that no mistake is introduced by the preprocessing of the data.

        Raises:
            Exception: If an incoherency is detected, the argument of the error indicate where the error is detected
        """

        raw_pheno_df = pd.read_csv("../Data/BBBDL_pheno_20000bbb_6traits_processed.csv", index_col=1)
        raw_pheno_df = raw_pheno_df.drop(["col_1"], axis = 1)   

        for pheno in self.phenotypes.keys():
            # Drop the na induced by the merge of the differents series
            pheno_data = self.phenotypes[pheno]

            # Check that shapes are coherant with what expected
            SNP_data = self.get_SNP(pheno)
            if pheno_data.shape[0] != SNP_data.shape[0]:
                raise Exception(f"The number of sample of phenotype {pheno_data.shape[0]} and SNP data {SNP_data.shape[0]} are not coherant for {pheno}.")
            
            for i in range(SNP_data.shape[0]):
                # First check that the phenotype linked to the id is the same than in the raw file
                index = pheno_data.index[i]
                if pheno_data.iloc[i] != raw_pheno_df[pheno].loc[index]:
                    raise Exception(f"Phenotype {pheno} linked to id {index} ({pheno_data.iloc[i]}) isn't the same than the one in original data ({raw_pheno_df[pheno].loc[index]}).")

                # Then check that the SNP_data is the same than the one given by the function
                index_in_list = int(index.split("_")[1]) - 1
                if np.array(self.__input[index_in_list]).all() != np.array(SNP_data[i]).all():
                    raise Exception(f"SNP data in phenotype {pheno} for id {index} isn't the same than the original data.")

    
    
    # TODO: update these function for pytorch usage
    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        return self.input[idx], self.pheno.iloc[idx]