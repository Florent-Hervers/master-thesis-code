import wandb
import torch
import numpy as np
import random
import torch.utils.data as data

from MLP import MLP
from dataset import SNPmarkersDataset
from utils import train_DL_model


def main():
    torch.manual_seed(2307)
    np.random.seed(7032)
    random.seed(3072)
    g = torch.Generator()
    g.manual_seed(7230)

    BATCH_SIZE = 256
    DROPOUT = 0.25
    N_LAYERS = 10
    HIDDEN_NODES = [1024, 1024, 1024, 1024, 728, 512, 512, 512, 512]
    N_EPOCHS = 200
    MODEL_NAME = "Deep_MLP"

    run = wandb.init(
        project = "TFE",
        config = {
            "model_name": MODEL_NAME,
            "batch size": BATCH_SIZE,
            "dropout": DROPOUT,
            "nb layers": N_LAYERS,
            "hidden layers size": HIDDEN_NODES,
            "nb epochs": N_EPOCHS,
        },
        tags = ["tuning"],
    )
    run.name = f"{MODEL_NAME} with lr = {wandb.config.learning_rate:.3E}"

    train_dataset = SNPmarkersDataset(mode = "train", skip_check=True, normalize=True)
    validation_dataset = SNPmarkersDataset(mode = "validation", skip_check=True, normalize=True)
    phenotype = list(train_dataset.phenotypes.keys())[0]

    train_dataset.set_phenotypes = phenotype

    # Define function and seed to fix the loading via the dataloader (from https://pytorch.org/docs/stable/notes/randomness.html#pytorch)
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4, worker_init_fn=seed_worker, generator=g)
    
    validation_dataset.set_phenotypes = phenotype
    validation_dataloader = data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers = 4, worker_init_fn=seed_worker, generator=g)

    model = MLP(nlayers=N_LAYERS, hidden_nodes= HIDDEN_NODES, dropout= DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    criterion = torch.nn.L1Loss()

    train_DL_model(
        model,
        optimizer,
        train_dataloader,
        validation_dataloader,
        N_EPOCHS,
        criterion,
        phenotype=phenotype,
        early_stop_n_epoch= 10,
        early_stop_threshold=0.9,
    )

if __name__ == "__main__":
    sweep_config = {
        "name": "Deep_MLP ep_res lr tuning",
        "method": "grid",
        "metric": {
            "goal": "maximize",
            "name": "correlation ep_res.max"
        },
        "parameters": {
            "learning_rate": {
                # np.linspace(1e-3,1e-6,20)
                "values": [1.00000000e-03, 9.47421053e-04, 8.94842105e-04, 8.42263158e-04,
        7.89684211e-04, 7.37105263e-04, 6.84526316e-04, 6.31947368e-04,
        5.79368421e-04, 5.26789474e-04, 4.74210526e-04, 4.21631579e-04,
        3.69052632e-04, 3.16473684e-04, 2.63894737e-04, 2.11315789e-04,
        1.58736842e-04, 1.06157895e-04, 5.35789474e-05, 1.00000000e-06]
            }
        },
    }
    sweep_id = wandb.sweep(sweep=sweep_config, project="TFE")
    wandb.agent(sweep_id, function=main)