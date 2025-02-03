import torch
import torch.utils.data as data
from dataset import SNPmarkersDataset
import wandb
from utils import train_DL_model, list_of_strings
import numpy as np
import random
from torch.utils.data import Dataset
from sklearn.feature_selection import mutual_info_regression
from argparse import ArgumentParser
import json
from hydra import compose,initialize
from omegaconf import OmegaConf
from Models.GPTransformer import GPTransformer

class SNPResidualDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]


def convert_categorical_to_frequency(data, path = "gptranformer_embedding_data.json"):
    with open(path,"r") as f:
        freq_data = json.load(f)
    
    results = []
    for sample in data:
        func = lambda t: [freq_data[str(t[0])]["p"]**2, 2*freq_data[str(t[0])]["p"]*freq_data[str(t[0])]["q"],freq_data[str(t[0])]["q"]**2].__getitem__(t[1])
        results.append(list(map(func, enumerate(sample))))
    return np.array(results, dtype=np.float32)

def main():
    torch.manual_seed(2307)
    np.random.seed(7032)
    random.seed(3072)
    g = torch.Generator()
    g.manual_seed(7230)


    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--encoding",
        choices=["categorial", "frequency"],
        required=True,
        help="Encoding of the input of the model to use (see paper for more detail)"
    )
    parser.add_argument(
        "--template",
        "-t",
        required=True,
        type=str, 
        help="Name of the file (without file extention) to use for the training (should be found in configs/template)"
    )
    parser.add_argument("--wandb_run_name", "-w", required=False, type=str, help="String to use for the wandb run name")
    parser.add_argument("--phenotypes", "-p", required=True, type=list_of_strings, help="Phenotype(s) to perform the sweep (format example: ep_res,de_res,size_res)")
    
    args = parser.parse_args()

    with initialize(version_base=None, config_path="Configs/template"):
        cfg = compose(
            config_name= args.template,
        )

    BATCH_SIZE = int(cfg.batch_size)
    LEARNING_RATE = cfg.learning_rate
    N_EMBEDDING = int(cfg.n_embedding)
    N_HEADS = int(cfg.n_heads)
    N_LAYERS = int(cfg.n_layers)
    HIDDEN_NODES = int(cfg.hidden_nodes)
    N_EPOCHS = int(cfg.n_epochs)
    MUTUAL_INFO_THRESHOLD = cfg.mutual_info_threshold

    wandb.init(
        project = "TFE",
        config = OmegaConf.to_object(cfg),
        name = args.wandb_run_name,
        tags = ["debug"],
    )

    train_dataset = SNPmarkersDataset(mode = "train")
    validation_dataset = SNPmarkersDataset(mode = "validation")
    selected_phenotypes = args.phenotypes

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
        indexes = np.where(mi > MUTUAL_INFO_THRESHOLD)[0]
        print(f"Nb of selected features: {len(indexes)}")

        if args.encoding == "categorical":
            # - 1 is used to shoft the [0,1,2] range to the [-1,0,1] used in the paper  
            train_dataset = SNPResidualDataset(X_train[indexes].to_numpy(dtype=np.float32) - 1, y_train.to_numpy(dtype=np.float32))
            validation_dataset = SNPResidualDataset(X_val[indexes].to_numpy(dtype=np.float32) - 1, y_val.to_numpy(dtype=np.float32))
        elif args.encoding == "frequency":
            train_dataset = SNPResidualDataset(convert_categorical_to_frequency(X_train[indexes].to_numpy()), y_train.to_numpy(dtype=np.float32))
            validation_dataset = SNPResidualDataset(convert_categorical_to_frequency(X_val[indexes].to_numpy()), y_val.to_numpy(dtype=np.float32))
        
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
            early_stop_n_epoch=cfg.early_stop_n_epoch,
            early_stop_threshold=cfg.early_stop_threshold,
        )
    
if __name__ == "__main__":
    main()