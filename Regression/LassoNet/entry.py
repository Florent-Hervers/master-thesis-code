bo_config_file = 'Regression/LassoNet/param_space.yaml'

from dataset import SNPmarkersDataset
import sys
import os
sys.path.append(os.path.abspath('Regression/LassoNet/'))
from bo import bo_search
import time


def main():
    start_time = time.time()
    selected_phenotype = "ep_res"

    train_dataset = SNPmarkersDataset(mode="train")
    train_dataset.set_phenotypes = selected_phenotype
    train_X = train_dataset.get_all_SNP()
    train_Y = train_dataset.phenotypes[selected_phenotype]

    validation_dataset = SNPmarkersDataset(mode="validation")
    validation_dataset.set_phenotypes = selected_phenotype
    validation_X = validation_dataset.get_all_SNP()
    validation_Y = validation_dataset.phenotypes[selected_phenotype]

    corr, optim_max = bo_search((train_X, train_Y, validation_X, validation_Y), bo_config_file)
    print("-------------------------------------------------------------------------------------")
    print(f"Optimal correlation found: {corr}")
    print(f"Optimal parameters: {optim_max}")

    print(f"Computation finished in {int((time.time() - start_time) // 3600)}h {int(((time.time() - start_time) % 3600) // 60)}m {int((time.time() - start_time) % 60)}s")


if __name__ == "__main__":
    main()