import torch
import wandb
from Models.ResGS import ResGSModel
from utils import get_clean_config, get_default_config_parser, print_elapsed_time, list_of_strings, train_from_config
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from torch.utils.data import Dataset
import numpy as np
import time
from argparse import ArgumentParser
from hydra import compose, initialize
from hydra.utils import instantiate

class SNPResidualDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


def train_on_residuals(phenotype: str, cfg):
    """Wrapper on the train_from_config that trains on the residual from the optimal ridge model.

    Args:
        phenotype (str): phenotype on which the model should be trained on (should be a key of SNPmarkersDataset.phenotypes).
        run_cfg (DictConfig): hydra config file fetched using the compose API. 
    """

    train_dataset = instantiate(cfg.data.train_dataset)
    validation_dataset = instantiate(cfg.data.validation_dataset)

    # Best ridge hyperparameters for all the phenotypes
    hp = {
        "ep_res": {"lambda": 55600},
        "de_res": {"lambda": 44500},
        "FESSEp_res": {"lambda": 26250},
        "FESSEa_res": {"lambda": 34000},
        "size_res": {"lambda": 20900},
        "MUSC_res": {"lambda": 23950},
    }
    
    start_time = time.time()
    train_dataset.set_phenotypes = phenotype
    validation_dataset.set_phenotypes = phenotype

    X_train = train_dataset.get_all_SNP()
    y_train = train_dataset.phenotypes[phenotype]

    X_val = validation_dataset.get_all_SNP()
    y_val = validation_dataset.phenotypes[phenotype]
    
    #Traditional model
    global bestTraditionalModel
    r_max = 0  # Record the maximum Pearson value
    for model_str in ["Ridge", "support vector machine", "RandomForest", "GradientBoostingRegressor"]:
        if model_str == "Ridge":
            model = Ridge(alpha= hp[phenotype]["lambda"])
        elif model_str == "support vector machine":
            continue
            model = SVR()
        elif model_str == "RandomForest":
            continue
            model = RandomForestRegressor(n_jobs=-1, random_state=2307)
        elif model_str == "GradientBoostingRegressor":
            continue
            model = GradientBoostingRegressor(random_state=2307)
        model.fit(X_train, y_train)
        y_pre = model.predict(X_val)
        r = pearsonr(y_pre, y_val).statistic
        print(f"Correlation of {r} obtained for model {model_str} in {print_elapsed_time(start_time)}.")
        if r > r_max:
            r_max = r
            bestTraditionalModel = model
        
    print(f"The best model for the phenotype {phenotype} is thus {bestTraditionalModel}")
    print(f"#########################################################################################################")

    y_train_pre = bestTraditionalModel.predict(X_train)
    y_pre = bestTraditionalModel.predict(X_val)

    print(f"Sample of original y_train: {y_train[0:10]}")
    print(f"Sample of original y_val: {y_val[0:10]}")
    
    y_train = y_train - y_train_pre
    y_val = y_val - y_pre

    y_train = y_train.to_numpy(dtype=np.float32)
    y_val = y_val.to_numpy(dtype=np.float32)

    """
    # Normalize phenotypes in order to have comparable distribution during the validation and during the training
    mean_val = y_val.mean()
    std_val = y_val.std()
    y_train = (y_train - y_train.mean()) / y_train.std()
    y_val = (y_val - y_val.mean()) / y_val.std()
    """

    print(f"Sample of residual y_train: {y_train[0:10]}")
    print(f"Sample of residual y_val: {y_val[0:10]}")
    
    residual_train_dataset = SNPResidualDataset(X_train.to_numpy(dtype=np.float32), y_train)
    residual_validation_dataset = SNPResidualDataset(X_val.to_numpy(dtype=np.float32), y_val)

    train_from_config(
        phenotype,
        cfg,
        residual_train_dataset,
        residual_validation_dataset,
        initial_phenotype = y_pre,
    )

if __name__ == "__main__":
    parser = get_default_config_parser()
    
    args = parser.parse_args()
    
    with initialize(version_base=None, config_path="Configs"):
        cfg = compose(
            config_name="default",
            overrides=[f"model_config={args.model}", f"data={args.data}", f"train_function_config={args.train_function}"],
        )

    if cfg.train_function_config.log_wandb:
        wandb.init(
            project = "TFE",
            config = get_clean_config(cfg),
            name = args.wandb_run_name,
            tags = ["debug"],
        )
    for phenotype in args.phenotypes:
        train_on_residuals(phenotype, cfg)