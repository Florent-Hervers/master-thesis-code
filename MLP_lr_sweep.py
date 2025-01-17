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

    N_LAYERS = 10
    N_EPOCHS = 200
    EARLY_STOP_THRESHOLD = 0.9
    EARLY_STOP_N_EPOCH = 10
    MODEL_NAME = "Deep_MLP"

    wandb.init(
        config = {
            "model_name": MODEL_NAME,
            "nb layers": N_LAYERS,
            "nb epochs": N_EPOCHS,
            "early stop threshold": EARLY_STOP_THRESHOLD,
            "early stop nb epoch": EARLY_STOP_N_EPOCH,
        },
        tags = ["tuning"],
    )
    BATCH_SIZE = wandb.config.batch_size
    DROPOUT = wandb.config.dropout
    HIDDEN_NODES = [
        wandb.config.hidden_layer_size,
        wandb.config.hidden_layer_size,
        wandb.config.hidden_layer_size,
        wandb.config.hidden_layer_size,
        wandb.config.hidden_layer_size - (wandb.config.hidden_layer_size // 4),
        wandb.config.hidden_layer_size // 2,
        wandb.config.hidden_layer_size // 2,
        wandb.config.hidden_layer_size // 2,
        wandb.config.hidden_layer_size // 2,
    ]

    train_dataset = SNPmarkersDataset(mode = "train", skip_check=True)
    validation_dataset = SNPmarkersDataset(mode = "validation", skip_check=True)
    phenotype = wandb.config.phenotype

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
        early_stop_n_epoch=EARLY_STOP_N_EPOCH,
        early_stop_threshold=EARLY_STOP_THRESHOLD,
    )

if __name__ == "__main__":
    for phenotype in ["ep_res", "size_res", "de_res"]:
        sweep_config = {
            "name": f"Deep_MLP {phenotype} full hyperparameter tuning",
            "method": "bayes",
            "metric": {
                "goal": "maximize",
                "name": "correlation ep_res.max"
            },
            "parameters": {
                "phenotype": {
                    "value": phenotype
                },
                "learning_rate": {
                    # np.linspace(1e-3,1e-6,10)
                    "values": [1.00e-03, 8.89e-04, 7.78e-04, 6.67e-04, 5.56e-04, 4.45e-04,
        3.34e-04, 2.23e-04, 1.12e-04, 1.00e-06]
                },
                #list(map(lambda v: 2**v, range(5,14))
                "hidden_layer_size": {
                    "values": [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
                },
                #np.arange(0.05,0.55,0.05)
                "dropout": {
                    "values": [0.05, 0.1 , 0.15, 0.2 , 0.25, 0.3 , 0.35, 0.4 , 0.45, 0.5 ]
                },
                #list(map(lambda v: 2**v, range(3,10))
                "batch_size": {
                    "values": [8, 16, 32, 64, 128, 256, 512, 1024]
                }
            },
            "run_cap": 75,
        }
        sweep_id = wandb.sweep(sweep=sweep_config, project="TFE")
        wandb.agent(sweep_id, function=main)