import torch
from torch import nn
import torch.utils.data as data
from dataset import SNPmarkersDataset
import wandb
from utils import train_DL_model
import numpy as np
import random
from torch.utils.data import Dataset
from sklearn.feature_selection import mutual_info_regression


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, n_hidden, n_heads):
        super(TransformerBlock, self).__init__()

        self.multihead = nn.MultiheadAttention(embedding_size, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.fc1 = nn.Linear(embedding_size, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
    
    def forward(self, x):
        y, _ = self.multihead(x,x,x)
        y = self.norm1(x + y)
        z = self.fc1(y)
        z = self.fc2(self.relu(z))
        return self.norm2(y + z)
    
class GPTransformer(nn.Module):
    def __init__(self,  n_features, embedding_size, n_hidden, n_heads, n_blocks):
        super(GPTransformer, self).__init__()
        self.n_features = n_features
        self.embedding_size = embedding_size
        self.embedding = nn.Linear(n_features, n_features * embedding_size) #nn.Embedding(3, embedding_size)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embedding_size, n_hidden, n_heads) for _ in range(n_blocks)]
        )
        self.output = nn.Linear(embedding_size * n_features, 1)
    
    def forward(self, x):
        x = self.embedding(x) #(x.int())
        x = x.view((x.shape[0], self.n_features, self.embedding_size))
        x = self.transformer(x)
        return self.output(x.view(x.shape[0], -1))


class SNPResidualDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    

def main():
    torch.manual_seed(2307)
    np.random.seed(7032)
    random.seed(3072)
    g = torch.Generator()
    g.manual_seed(7230)

    BATCH_SIZE = 64
    LEARNING_RATE = 5e-4
    DROPOUT = 0
    N_EMBEDDING = 8
    N_HEADS = 2
    N_LAYERS = 2
    HIDDEN_NODES = 256
    N_EPOCHS = 200
    MUTUAL_INFO_THRESHOLD = 0.02
    MODEL_NAME = "GPTransformer"
    RUN_NAME = "First run GPTransformer"

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
            "nb attention heads": N_HEADS,
            "embedding size": N_EMBEDDING

        },
        name = RUN_NAME,
        tags = ["debug"],
    )

    train_dataset = SNPmarkersDataset(mode = "train", normalize=False)
    validation_dataset = SNPmarkersDataset(mode = "validation", normalize=False)
    selected_phenotypes = list(train_dataset.phenotypes.keys())

    for phenotype in selected_phenotypes:
        mi = np.zeros(36304)
        modes = ["train", "validation", "test"]
        X_train = []
        y_train = []
        X_val = []
        y_val = []
        for mode in modes:
            dataset = SNPmarkersDataset(mode = mode, skip_check=True)
            dataset.set_phenotypes = phenotype
            
            X = dataset.get_all_SNP()
            y = dataset.phenotypes[phenotype]

            # Save the results to avoid fetching two times the sames values later on
            if mode == "train":
                X_train = X
                y_train = y
            if mode == "validation":
                X_val = X
                y_val = y

            mi += mutual_info_regression(X,y, n_jobs=-1, discrete_features=True, random_state=2307)

        # Divide the number of modes to obtain the average mutual information
        mi /= len(modes)
        indexes = indexes = np.where(mi > MUTUAL_INFO_THRESHOLD)[0]
        print(f"Nb of selected features: {len(indexes)}")

        # - 1 is used to shoft the [0,1,2] range to the [-1,0,1] used in the paper  
        train_dataset = SNPResidualDataset(X_train[indexes].to_numpy(dtype=np.float32) - 1, y_train.to_numpy(dtype=np.float32))
        validation_dataset = SNPResidualDataset(X_val[indexes].to_numpy(dtype=np.float32) - 1, y_val.to_numpy(dtype=np.float32))
        
        # Define function and seed to fix the loading via the dataloader (from https://pytorch.org/docs/stable/notes/randomness.html#pytorch)
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        train_dataloader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers = 4, worker_init_fn=seed_worker, generator=g)
        validation_dataloader = data.DataLoader(validation_dataset, batch_size=BATCH_SIZE, num_workers = 4, worker_init_fn=seed_worker, generator=g)

        model = GPTransformer(
            n_features=len(indexes),
            embedding_size=N_EMBEDDING, 
            n_hidden=HIDDEN_NODES,
            n_heads=N_HEADS,
            n_blocks=N_LAYERS
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = torch.nn.L1Loss()

        train_DL_model(
            model,
            optimizer,
            train_dataloader,
            validation_dataloader,
            N_EPOCHS,
            criterion,
            phenotype=phenotype,
        )
    
if __name__ == "__main__":
    main()