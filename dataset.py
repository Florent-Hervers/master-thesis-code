from bed_reader import open_bed
import pandas as pd
import numpy as np
from typing import List
from torch.utils.data import Dataset
import torch
import os
import time

class SNPmarkersDataset(Dataset):
    """Create a class following the pytorch template to manage the data. \
    Note: SNP data should be fetched using the `get_all_SNP()` function.

    Attributes:
        mode (str, optional): type of the dataset represented. Only 'train' or 'test' or 'validation' are accepted values. \
        Modifying this value will not change the type of dataset as this variable is designed to be "read-only". Defaults to "train".
        phenotypes (dict[str, pd.Series]): contain the phenotypes data for the given mode. Keys are phenotypes label used in the original phenotype files. \
        Phenotypes are exposed as an attribute to ease the data analysis while SNP_array can only be accessed via `get_all_SNP` or `__getItem` methods.
        set_phenotypes(str | list[str]): Default to None. Should be definied before using the dataset class by the user. Accepted string are the keys of ´self.phenotypes´
        pheno_std(dict[str, float]): Store the standard deviasion of the selected phenotypes. Keys are phenotypes label used in the original phenotype files. \
        Return a empty dictonary if set_phenotypes isn't set.
    """
    
    def __init__(
        self, mode = "train", 
        dir_path = "../Data", 
        skip_check = False, 
        bed_filename = "BBBDL_BBB2023_MD.bed", 
        pheno_filename = "BBBDL_pheno_20000bbb_6traits_processed.csv", 
        mask_pheno_filename = "BBBDL_pheno_2023bbb_0twins_6traits_mask_processed.csv",
        date_filename = "BBBDL_pedi_full_list.txt",
        normalize = True,
    ):
        """Create a class following the pytorch template.

        Args:
            mode (str, optional): type of the data stored. Only train, validation, test and local_train are accepted values. Defaults to "train".
            dir_path (str, optional): Argument used to specify the path to the data directory. Defaults to "../Data/".
            skip_check (bool, optional): Option to skip the checking used to speed object creation when testing. Defaults to False.
            bed_filename (str, optional): name of the bed file containing the snp array data. Defaults to "BBBDL_BBB2023_MD.bed".
            pheno_filename (str, optional): File containing the phenotypes in csv format. The file must have an unused column as first column. \
            The second one will be the one used as index. A header containing the name of each column must be provided with the first empty column named "col_1". \
            Keys to access phenotypes will be infered from this header. Defaults to "BBBDL_pheno_20000bbb_6traits_processed.csv".
            mask_pheno_filename (str,optional): File where phenotypes of all samples that must be used for the test dataset are masked (ie set to NA). \
            This file must follow the same format than the one given in pheno_filename.
            date_filename (str | None, optional): File used to sort samples by their birth dates. The file must be a csv file with three columns separated by tabs \
            The first columns must contain the id of the animal and the birth date must be the thrid and last columns formated in yyyymmdd (where y=year, m=month, d=day). \
            The second columns will be used. Note that the all masked samples must be born after the ones in the training/validation set for the sorting to be relevant. \
            Setting this parameter to None will prevent the sorting.
            normalize (bool): define if the phenotypes outputed by the function getitem should be normalized. Defaults to True.

        Raises:
            AttributeError: if the mode doesn't respect the described format.
            IOError: if an error occured when opening one of the files
        """
        # Define the number of samples used for the validation set
        self._VALIDATION_SIZE = 1000
        # Define variable to remember the phenotype to use in the different functions
        self._wantedPhenotypes: str | List[str] = None
        # Define the path to the phenotypes file for the check_data function
        self._pheno_filepath = os.path.join(dir_path, pheno_filename)

        self.normalize = normalize

        try:
            bed_file_data = open_bed(os.path.join(dir_path, bed_filename)).read(dtype="int8", num_threads= 8)
        except Exception as e:
            raise IOError(f"The following error occured when trying to read the bed file: {e.args}")
        
        try:
            pheno_masked_df = pd.read_csv(os.path.join(dir_path, mask_pheno_filename), index_col= 1).drop(["col_1"], axis = 1)
            self._snp = pd.DataFrame(bed_file_data, index=pheno_masked_df.index)
        except Exception as e:
            raise IOError(f"The following error occured when trying to read the masked phenotype file: {e.args}")
        
        valid_modes = set(["train", "validation", "test", "local_train"])
        if mode not in valid_modes:
            raise AttributeError(f"the mode argument must be a value of {valid_modes}!")
        self.mode = mode

        if date_filename is not None:
            sorted_index = pd.read_csv(
                os.path.join(dir_path, date_filename), 
                sep = "\t", 
                names=["id", 'sex', "birth_date"], 
                index_col = 0, 
                converters= {"birth_date": lambda date: time.strptime(str(date),"%Y%m%d")}
            ).sort_values("birth_date")

            train_indexes = pheno_masked_df.dropna(how="all").index
            test_indexes = list(set(pheno_masked_df.index) -  set(train_indexes))
            assert max(sorted_index.loc[train_indexes, "birth_date"]) < min(sorted_index.loc[test_indexes, "birth_date"]), "All samples non masked should be borned before the ones used in the test set"

            pheno_masked_df = pheno_masked_df.reindex(sorted_index.index)

        # Removal of the masked samples is done after the sorting to be able to use the id of all sample within the dataframe
        pheno_masked_df = pheno_masked_df.dropna(how="all")

        self.phenotypes = {}        
        if mode == "local_train":
            for pheno in pheno_masked_df.columns:
                self.phenotypes[pheno] = pheno_masked_df[pheno].dropna().iloc[:1000]

        elif mode == "train":   
            for pheno in pheno_masked_df.columns:
                self.phenotypes[pheno] = pheno_masked_df[pheno].dropna().iloc[:-self._VALIDATION_SIZE]
            
        elif mode == "validation":
            for pheno in pheno_masked_df.columns:
                self.phenotypes[pheno] = pheno_masked_df[pheno].dropna().iloc[-self._VALIDATION_SIZE:]

        elif mode == "test":
            try:
                all_pheno_df = pd.read_csv(self._pheno_filepath, index_col=1)
                all_test_samples = all_pheno_df.loc[(~ all_pheno_df.index.isin(pheno_masked_df.index)).tolist()].drop(["col_1"], axis = 1)
                for pheno in pheno_masked_df.columns:
                    self.phenotypes[pheno] = all_test_samples[pheno].dropna()
            except Exception as e:
                raise IOError(f"The following error occured when trying to open the phenotype file: {e.args}")
        
        self.pheno_std = {}

        if not skip_check:
            try:
                self.check_data()
            except Exception as e:
                raise e
    
    def get_all_SNP(self):
        """ Retruns all snp arrays that have values for the phenotypes defined via `set_phenotypes proprety`

        Raises:
            Exception: If the proprety `set_phenotypes` isn't set.
        
        Returns:
            pd.Series: snp array for every sample in the mode (with their corresponding id from the original file)
        """
        if self._wantedPhenotypes == None:
            raise Exception("The proprety set_phenotypes must be set before using the dataset")

        indexes = pd.DataFrame.from_dict(self.phenotypes)[self._wantedPhenotypes].dropna(how="all").index
        
        # Need to sort the index according to the phenotypes after selecting the index to have the correct relation snp_data-pheotype
        # Cause by the sorting of the id when creating the indexes dataframe that doesn't follow the one used in pheno files
        if type(self._wantedPhenotypes) == str:
            return self._snp.loc[indexes].reindex(self.phenotypes[self._wantedPhenotypes].index)
        else:
            return self._snp.loc[indexes].reindex(self.phenotypes[self._wantedPhenotypes[0]].index)

        
    def check_data(self):
        """Perform some verifications on the data (of the current object) to be sure that no mistake is introduced by the preprocessing of the data.

        Raises:
            Exception: If an incoherency is detected, the argument of the error indicate where the error is detected
        """

        # Check that data is free of missing values
        classes, counts = np.unique(self._snp, return_counts=True)
        if classes.all() != np.array([0, 1, 2]).all():
            raise Exception(f"There are {counts[np.where(classes == -127)[0][0]]} missing values in the data!")

        try: 
            raw_pheno_df = pd.read_csv(self._pheno_filepath, index_col=1)
        except Exception as e:
                raise IOError(f"The following error occured when trying to open the phenotype file: {e.args}")  

        for pheno in self.phenotypes.keys():
            for index in self.phenotypes[pheno].index:
                if index not in self._snp.index:
                    raise Exception(f"No SNP array found for index {index}")
                if raw_pheno_df[pheno].loc[index] != self.phenotypes[pheno].loc[index]:
                    raise Exception(f"Uncoherant value in dataset for the index {index}, phenotype {pheno}")
            
            if (self.mode == "validation"):
                if len(self.phenotypes[pheno].index) != self._VALIDATION_SIZE:
                    raise Exception(f"There isn't {self._VALIDATION_SIZE} samples in the validation set but rather {len(self.phenotypes[pheno].index)}")

    def __len__(self):
        """Return the length of the dataset for the selected mode and phenotype

        Raises:
            Exception: If the proprety `set_phenotypes` isn't set.

        Returns:
            int: length of the dataset
        """
        if self._wantedPhenotypes == None:
            raise Exception("The proprety set_phenotypes must be set before using the dataset")

        if type(self._wantedPhenotypes) == str:
            return len(self.phenotypes[self._wantedPhenotypes].index)
        else:
            return len(self.phenotypes[self._wantedPhenotypes[0]].index)

    def __getitem__(self, idx):
        """Return the snp data and selected phenotype (via the proprety `set_phenotypes`) given an index. \
        The value of the phenotype data returned will be normalized.

        Args:
            idx (int): index in the dataset.

        Raises:
            Exception: If the proprety `set_phenotypes` isn't set.

        Returns:
            (pd.Series, float | dict): first item is the snp array, the second one type depend on the number of phenoype wanted. If more than one return a dict, otherwise a float
        """
        if self._wantedPhenotypes == None:
            raise Exception("The proprety set_phenotypes must be set before using the dataset")
        
        if type(self._wantedPhenotypes) == str:
            index = self.phenotypes[self._wantedPhenotypes].index[idx]
        else:
            index = self.phenotypes[self._wantedPhenotypes[0]].index[idx]
        
        # Check that the label exist, if not insert nan.
        if type(self._wantedPhenotypes) == list:
            phenotype_data = {}

            for pheno in self._wantedPhenotypes:
                if self.normalize:
                    phenotype_data[pheno] = self.phenotypes[pheno].loc[index] / self.pheno_std[self._wantedPhenotypes]
                else:
                    phenotype_data[pheno] = self.phenotypes[pheno].loc[index]

            return torch.tensor(self._snp.loc[index], dtype=torch.float), phenotype_data
        else:
            target = self.phenotypes[self._wantedPhenotypes].loc[index]
            if self.normalize:
                target /= self.pheno_std[self._wantedPhenotypes]
            return torch.tensor(self._snp.loc[index], dtype=torch.float) , target
    
    @property
    def set_phenotypes(self):
        """Define the phenotype(s) to returns when using getItem functions. Note that the inputed string should be the name of the columns of the original dataframe.
        If several phenotypes are entered, only combinaision allowed are the one where all samples in the dataset have a value for all selected phenotypes in order to avoid introducing nan in the loss.

        Returns:
            str/List[str]: the name of the phenotype returned (or the list of phenotypes if user selected more than one)
        """
        return self._wantedPhenotypes

    @set_phenotypes.setter
    def set_phenotypes(self, value):
        if type(value) != str and type(value) != list:
            raise Exception("This variable expect either a string or a list of strings")
        
        # Convert the list to str if only one item in the list to have an appropriate behaviour in getitem
        if type(value) == list and len(value) == 1 and type(value[0]) == str:
            value =  value[0]
        
        iterable = value
        if type(value) == str:
            iterable = [value]
        
        for item in iterable:
            if type(item) != str:
                raise Exception("This variable expect either a string or a list of strings")
            if item not in self.phenotypes.keys():
                raise Exception(f"The phenotypes {item} isn't found in the phenotypes")
        
        # Check that the given phenotypes are compatible to avoid nan in the loss
        if type(value) == list:
            df = pd.DataFrame.from_dict(self.phenotypes)[value].dropna(how='all')
            if df.shape != df.dropna(how='any').shape:
                raise Exception("The given phenotypes aren't compatible as they aren't definied for every phenotype!")
 
        self._wantedPhenotypes = value
        
        # Also recompute the standard deviation for the selected phenotypes
        for pheno in iterable:
            self.pheno_std[pheno] = np.std(self.phenotypes[pheno])

    @set_phenotypes.deleter
    def set_phenotypes(self):
        del self._wantedPhenotypes