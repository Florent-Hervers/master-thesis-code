import torch
from torch import nn
import torch.utils.data as data
from dataset import SNPmarkersDataset
import torch.nn.functional as F
import wandb
import numpy as np
from scipy.stats import pearsonr
from utils import format_batch

class MLP(torch.nn.Module):
    def __init__(self, nlayers: int = 1, hidden_nodes: list[int] = [], dropout: float = 0):
        super(MLP, self).__init__()
        
        if dropout < 0 or dropout >= 1:
            raise AttributeError("The dropout must be between 0 and 1")

        if nlayers < 1:
            raise AttributeError("The number of layers must be greater or equal than one !")
        
        if len(hidden_nodes) != nlayers - 1:
            raise AttributeError(f"Not enough hidden_nodes given, expected a list of length {nlayers - 1} but got one of {len(hidden_nodes)}")

        hidden_nodes.insert(0, 36304)
        hidden_nodes.append(4)

        self.model = nn.Sequential(*[LinearBlock(hidden_nodes[i], hidden_nodes[i + 1], dropout=dropout) for i in range(nlayers - 1)])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_nodes[-2], hidden_nodes[-1])

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
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-3
    DROPOUT = 0.25
    N_LAYERS = 10
    HIDDEN_NODES = [1024, 1024, 1024, 1024, 768 ,512, 512, 512, 512]
    N_EPOCHS = 200
    SCHEDULER_STEP_SIZE = 20
    SCHEDULER_REDUCE_RATIO = 0.5
    MODEL_NAME = "Deep_MLP_all"
    RUN_NAME = "Try deep_MLP_all with normalization"

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
    
    selected_phenotypes = ["ep_res", "de_res", "FESSEp_res", "FESSEa_res"]
    train_dataset = SNPmarkersDataset(mode = "train", normalize=True)
    train_dataset.set_phenotypes = selected_phenotypes
    train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4)
    
    validation_dataset = SNPmarkersDataset(mode = "validation", normalize=True)
    validation_dataset.set_phenotypes = selected_phenotypes
    validation_dataloader = data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers = 4)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    model = MLP(nlayers=N_LAYERS, hidden_nodes= HIDDEN_NODES, dropout= DROPOUT)
    print(f"Model architecture : \n {model}")
    print(f"Numbers of parameters: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = SCHEDULER_STEP_SIZE, gamma = SCHEDULER_REDUCE_RATIO)
    criterion = torch.nn.L1Loss()
    model.to(device)
    for epoch in range(N_EPOCHS):
        train_loss = []
        model.train()
        for x,y in train_dataloader:
            x,y = x.to(device), format_batch(y).to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            train_loss.append(loss.cpu().detach())
            loss.backward()
            optimizer.step()
        
        wandb.log({
                "epoch": epoch, 
                "train_loss": np.array(train_loss).mean()
            }
        )

        print(f"Finished training for epoch {epoch}.")

        val_loss = []
        predicted = []
        target = []
        with torch.no_grad():
            model.eval()
            for x,y in validation_dataloader:
                x,y = x.to(device), format_batch(y).to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss.append(loss.cpu().detach())
                if len(predicted) == 0:
                    predicted = output.cpu().detach()
                    target = y.cpu().detach()
                else:
                    predicted = np.concatenate((predicted, output.cpu().detach()), axis = 0)
                    target = np.concatenate((target, y.cpu().detach()), axis = 0)
        
        scheduler.step()
        wandb.log({
            "epoch": epoch, 
            "validation_loss all phenotypes": np.array(val_loss).mean(),
        })
        
        for k, pheno in enumerate(selected_phenotypes):
            wandb.log({
                "epoch": epoch,
                f"correlation {pheno}": pearsonr(predicted[:, k], target[:, k]).statistic,
            })
            
        print(f"Validation step for epoch {epoch} finished!")
    
    # TODO add model saving step every x epoch 

if __name__ == "__main__":
    main()