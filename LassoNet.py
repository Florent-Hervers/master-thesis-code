from dataset import SNPmarkersDataset
from Regression.LassoNet.lassonet.lassonet.interfaces import LassoNetRegressor
from torch.optim import Adam
from functools import partial
import time

def main():
    train_dataset = SNPmarkersDataset(mode="train", skip_check=True)
    train_X = train_dataset.get_all_SNP("pheno_1")
    train_Y = train_dataset.phenotypes["pheno_1"]

    validation_dataset = SNPmarkersDataset(mode="validation", skip_check=True)
    validation_X = validation_dataset.get_all_SNP("pheno_1")
    validation_Y = validation_dataset.phenotypes["pheno_1"]

    print(f"Train X shape {train_X.shape}")
    print(f"Train Y shape {train_Y.shape}")
    print(f"Validation X shape {validation_X.shape}")
    print(f"Validation Y shape {validation_Y.shape}")

    start_time = time.time()

    model = LassoNetRegressor(
        hidden_dims=(1024,),
        batch_size=64,
        optim = partial(Adam, lr= 1e-3),
        lambda_start=0.01,
        device="cuda",
    )

    output = model.path(X=train_X, y=train_Y, X_val=validation_X, y_val=validation_Y, return_state_dicts=False)

    for item in output:
        print(item)

    print(f"Computation finished in {int((time.time() - start_time) // 3600)}h {int(((time.time() - start_time) % 3600) // 60)}m {int((time.time() - start_time) % 60)}s")

if __name__  == "__main__":
    main()