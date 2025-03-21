bo_config_file = 'param_space.yaml'

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import SNPmarkersDataset
from bo import bo_search
import time


def main():
    start_time = time.time()
    selected_phenotype = "ep_res"
    
    train_dataset = SNPmarkersDataset(mode="train", dir_path="../../Data")
    train_dataset.set_phenotypes = selected_phenotype
    train_X = train_dataset.get_all_SNP()
    train_Y = train_dataset.phenotypes[selected_phenotype]

    validation_dataset = SNPmarkersDataset(mode="validation", dir_path="../../Data")
    validation_dataset.set_phenotypes = selected_phenotype
    validation_X = validation_dataset.get_all_SNP()
    validation_Y = validation_dataset.phenotypes[selected_phenotype]

    bo_search((train_X, train_Y, validation_X, validation_Y), start_time, bo_config_filepath=bo_config_file)
    
if __name__ == "__main__":
    main()