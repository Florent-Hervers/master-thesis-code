import torch
import torch.utils.data as data
from dataset import SNPmarkersDataset
import wandb
import numpy as np
from utils import train_DL_model
import random
from Models.MLP import MLP

def main():
    torch.manual_seed(2307)
    np.random.seed(7032)
    random.seed(3072)
    g = torch.Generator()
    g.manual_seed(7230)

    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    DROPOUT = 0.25
    N_LAYERS = 10
    HIDDEN_NODES = [1024, 1024, 1024, 1024, 728, 512, 512, 512, 512]
    N_EPOCHS = 200
    SCHEDULER_STEP_SIZE = 20
    SCHEDULER_REDUCE_RATIO = 0.5
    MODEL_NAME = "Deep_MLP"
    RUN_NAME = "Deep MLP with normalization"

    wandb.init(
        project = "TFE",
        config = {
            "model_name": MODEL_NAME,
            "batch size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "dropout": DROPOUT,
            "nb layers": N_LAYERS,
            "hidden layers size": HIDDEN_NODES,
            "nb epochs": N_EPOCHS,
            "scheduler_reduce_ratio": SCHEDULER_REDUCE_RATIO,
            "scheduler_step_size": SCHEDULER_STEP_SIZE,
        },
        name = RUN_NAME,
        tags = ["debug"],
    )
    
    train_dataset = SNPmarkersDataset(mode = "train", skip_check=True, normalize=True)
    validation_dataset = SNPmarkersDataset(mode = "validation", skip_check=True, normalize=True)
    selected_phenotypes = list(train_dataset.phenotypes.keys())

    for phenotype in selected_phenotypes:
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
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = SCHEDULER_STEP_SIZE, gamma = SCHEDULER_REDUCE_RATIO)
        criterion = torch.nn.L1Loss()

        train_DL_model(
            model,
            optimizer,
            train_dataloader,
            validation_dataloader,
            N_EPOCHS,
            criterion,
            scheduler=scheduler,
            phenotype=phenotype,
            validation_std=validation_dataset.pheno_std[phenotype]
        )

if __name__ == "__main__":
    main()