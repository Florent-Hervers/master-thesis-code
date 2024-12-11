import torch
from torch import nn
import torch.utils.data as data
from dataset import SNPmarkersDataset
import torch.nn.functional as F
import wandb
import numpy as np
from utils import train_DL_model
import random

class MLP(torch.nn.Module):
    def __init__(self, nlayers: int = 1, hidden_nodes: list[int] = [], dropout: float = 0):
        super(MLP, self).__init__()
        
        if dropout < 0 or dropout >= 1:
            raise AttributeError("The dropout must be between 0 and 1")

        if nlayers < 1:
            raise AttributeError("The number of layers must be greater or equal than one !")
        
        if len(hidden_nodes) != nlayers - 1:
            raise AttributeError(f"Not enough hidden_nodes given, expected a list of length {nlayers - 1} but got one of {len(hidden_nodes)}")

        # Use a copy to avoid modifying the hyperparameter value for future runs
        hidden_nodes_model = hidden_nodes.copy()
        hidden_nodes_model.insert(0, 36304)
        hidden_nodes_model.append(1)

        self.model = nn.Sequential(*[LinearBlock(hidden_nodes_model[i], hidden_nodes_model[i + 1], dropout=dropout) for i in range(nlayers - 1)])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_nodes_model[-2], hidden_nodes_model[-1])

    def forward(self, x):
        return self.output_layer(self.dropout(self.model(x)))

class LinearBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout = 0):
        super(LinearBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=input_size, out_features=output_size)
    
    def forward(self,x):
        return F.relu(self.fc(x))

def main():
    torch.manual_seed(2307)
    np.random.seed(7032)
    random.seed(3072)
    g = torch.Generator()
    g.manual_seed(7230)

    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    DROPOUT = 0.25
    N_LAYERS = 2
    HIDDEN_NODES = [1024] # [1024, 1024, 1024, 1024, 728, 512, 512, 512, 512]
    N_EPOCHS = 200
    SCHEDULER_STEP_SIZE = 20
    SCHEDULER_REDUCE_RATIO = 0.5
    MODEL_NAME = "Shallow_MLP"
    RUN_NAME = "Rerun shallow MLP"

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
    
    train_dataset = SNPmarkersDataset(mode = "train", skip_check=True, normalize=False)
    validation_dataset = SNPmarkersDataset(mode = "validation", skip_check=True, normalize=False)
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
        )

if __name__ == "__main__":
    main()