import torch
import wandb
from ResGS import ResGSModel
from dataset import SNPmarkersDataset
from utils import train_DL_model, print_elapsed_time
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from torch.utils.data import Dataset
import numpy as np
import time


class SNPResidualDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


def main():
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    DROPOUT = 0
    N_LAYERS = 4
    N_EPOCHS = 100
    SCHEDULER_STEP_SIZE = 200
    SCHEDULER_REDUCE_RATIO = 1
    KERNEL_SIZE = 3
    CHANNEL_FACTOR1 = 4
    CHANNEL_FACTOR2 = 1.1
    NFILTERS = 64
    MODEL_NAME = "ResGS paper"
    RUN_NAME = "Run ResGS paper"

    wandb.init(
        project = "TFE",
        config = {
            "model_name": MODEL_NAME,
            "batch size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dropout": DROPOUT,
            "nb layers": N_LAYERS,
            "nb epochs": N_EPOCHS,
            "scheduler_reduce_ratio": SCHEDULER_REDUCE_RATIO,
            "scheduler_step_size": SCHEDULER_STEP_SIZE,
            "kernel size" : KERNEL_SIZE,
            "channel factor 1": CHANNEL_FACTOR1,
            "channel factor 2": CHANNEL_FACTOR2,
            "nfilters": NFILTERS
        },
        name = RUN_NAME,
        tags = ["debug"],
    )
    
    train_dataset = SNPmarkersDataset(mode = "train")
    validation_dataset = SNPmarkersDataset(mode = "validation")
    selected_phenotypes = list(train_dataset.phenotypes.keys())

    for phenotype in selected_phenotypes:
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
                model = Ridge(random_state=2307)
            elif model_str == "support vector machine":
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

        print(f"Sample of residual y_train: {y_train[0:10]}")
        print(f"Sample of residual y_val: {y_val[0:10]}")

        residual_train_dataset = SNPResidualDataset(X_train.to_numpy(dtype=np.float32), y_train.to_numpy(dtype=np.float32))
        residual_validation_dataset = SNPResidualDataset(X_val.to_numpy(dtype=np.float32), y_val.to_numpy(dtype=np.float32))

        train_dataloader = DataLoader(residual_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
        validation_dataloader = DataLoader(residual_validation_dataset, batch_size=BATCH_SIZE, num_workers = 4)

        model = ResGSModel(NFILTERS, KERNEL_SIZE, CHANNEL_FACTOR1, CHANNEL_FACTOR2, N_LAYERS)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criteron = torch.nn.L1Loss()
        
        train_DL_model(
            model=model,
            optimizer=optimizer,
            train_dataloader=train_dataloader,
            validation_dataloader=validation_dataloader,
            criterion=criteron,
            n_epoch=N_EPOCHS,
            phenotype=phenotype,
            initial_phenotype = y_pre,
        )

if __name__ == "__main__":
    main()