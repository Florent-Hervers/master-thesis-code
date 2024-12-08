import torch
from torch import nn
import torch.utils.data as data
from dataset import SNPmarkersDataset
import wandb
from utils import train_DL_model

class Conv1d_BN(nn.Module):
    def __init__(self, input_size, nb_filter, kernel_size, strides=1, padding = 1):
        super(Conv1d_BN, self).__init__()
        self.conv = nn.Conv1d(input_size, nb_filter, kernel_size, padding= padding, stride=strides)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(nb_filter)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class Res_Block(nn.Module):
    def __init__(self, input_size, nb_filter, kernel_size, strides=1):
        super(Res_Block, self).__init__()
        self.block = Conv1d_BN(input_size,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides)
    
    def forward(self, x):
        x = x + self.block(x)
        return x

class ResGSModel(nn.Module):

    def __init__(self, nFilter, _KERNEL_SIZE, CHANNEL_FACTOR1, CHANNEL_FACTOR2, nlayers = 8):
        super(ResGSModel, self).__init__()
        self.input_block1 = Res_Block(1, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
        self.input_block2 = Res_Block(nFilter, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1)
        nFilter1 = int(nFilter * CHANNEL_FACTOR1)

        self.layers = nn.Sequential(
            *[nn.Sequential( 
                Conv1d_BN(int(nFilter * CHANNEL_FACTOR2**(i-1)), nb_filter=nFilter1, kernel_size=_KERNEL_SIZE, strides=2), 
                Conv1d_BN(nFilter1, nb_filter=int(nFilter * CHANNEL_FACTOR2**i), kernel_size=1, strides=1, padding=0), 
                Res_Block(int(nFilter * CHANNEL_FACTOR2**i), nb_filter=int(nFilter * CHANNEL_FACTOR2**i), kernel_size=_KERNEL_SIZE, strides=1), 
                Res_Block(int(nFilter * CHANNEL_FACTOR2**i), nb_filter=int(nFilter * CHANNEL_FACTOR2**i), kernel_size=_KERNEL_SIZE, strides=1),
            )for i in range(1, nlayers + 1) ])

        self.output = nn.Sequential(
            Conv1d_BN(int(nFilter * CHANNEL_FACTOR2**nlayers), nb_filter= 6400 // (int(nFilter * CHANNEL_FACTOR2**nlayers)), kernel_size=1, strides=1, padding=0),
            nn.Flatten(),
            nn.Linear((6400 // (int(nFilter * CHANNEL_FACTOR2**nlayers))) * int(36304 / (2**nlayers)), 1)
        )

    def forward(self, x):
        # Set the number of channels to 1 as required by the conv1d layer
        x = x.view(x.shape[0], 1, x.shape[1])
        
        x = self.input_block1(x)
        x = self.input_block2(x)

        x = self.layers(x)

        x = self.output(x)
        return x


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