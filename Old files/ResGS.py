import torch
import torch.utils.data as data
from dataset import SNPmarkersDataset
import wandb
from utils import train_DL_model
from Models.ResGS import ResGSModel



def main():
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    DROPOUT = 0
    N_LAYERS = 4
    N_EPOCHS = 200
    SCHEDULER_STEP_SIZE = 200
    SCHEDULER_REDUCE_RATIO = 1
    KERNEL_SIZE = 3
    CHANNEL_FACTOR1 = 4
    CHANNEL_FACTOR2 = 1.1
    NFILTERS = 64
    MODEL_NAME = "ResGS"
    RUN_NAME = "Rerun ResGS on original dataset"

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
        train_dataset.set_phenotypes = phenotype
        validation_dataset.set_phenotypes = phenotype

        train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
        validation_dataloader = data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers = 4)

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
        )

if __name__ == "__main__":
    main()