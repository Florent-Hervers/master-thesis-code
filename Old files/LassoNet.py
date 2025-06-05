from dataset import SNPmarkersDataset
from Regression.LassoNet.lassonet.lassonet.interfaces import LassoNetRegressor
from torch.optim import Adam
from functools import partial
import time
import numpy as np

def main():
    selected_phenotype = "ep_res"

    train_dataset = SNPmarkersDataset(mode="train")
    train_dataset.set_phenotypes = selected_phenotype
    train_X = train_dataset.get_all_SNP()
    train_Y = train_dataset.phenotypes[selected_phenotype]

    validation_dataset = SNPmarkersDataset(mode="validation")
    validation_dataset.set_phenotypes = selected_phenotype
    validation_X = validation_dataset.get_all_SNP()
    validation_Y = validation_dataset.phenotypes[selected_phenotype]

    print(f"Train X shape {train_X.shape}")
    print(f"Train Y shape {train_Y.shape}")
    print(f"Validation X shape {validation_X.shape}")
    print(f"Validation Y shape {validation_Y.shape}")

    start_time = time.time()

    model = LassoNetRegressor(
        hidden_dims=(1024,),
        batch_size=64,
        optim = partial(Adam, lr= 1e-3),
        dropout = 0.25,
        tol = 0.995,
        patience=(100,10),
        M=0.5,
        lambda_seq = np.concatenate((np.arange(0.02,1.0,0.02), np.arange(1.0,10.0,0.1), np.arange(10.0,65.0,1))),
        device="cuda",
    )

    output = model.path(X=train_X, y=train_Y, X_val=validation_X, y_val=validation_Y, return_state_dicts=False)

    for item in output:
        print(item)

    print(f"Computation finished in {int((time.time() - start_time) // 3600)}h {int(((time.time() - start_time) % 3600) // 60)}m {int((time.time() - start_time) % 60)}s")

if __name__  == "__main__":
    main()