from dataset import SNPmarkersDataset
import wandb
from utils import train_from_config, list_of_strings, get_clean_config
import numpy as np
from torch.utils.data import Dataset
from sklearn.feature_selection import mutual_info_regression
from argparse import ArgumentParser
import json
from hydra import compose,initialize
from hydra.utils import instantiate
from Models.GPTransformer import EmbeddingType
import torch

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
    parser = ArgumentParser()
    parser.add_argument(
        "-e",
        "--encoding",
        choices=["categorial", "frequency", "one_hot", "learned"],
        required=True,
        help="Encoding of the input of the model to use (see paper for more detail)"
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        type=str, 
        help="Name of the file (without file extention) to define the model to train (should be found in configs/model_config)"
    )
    parser.add_argument("--wandb_run_name", "-w", required=False, type=str, help="String to use for the wandb run name")
    parser.add_argument("--phenotypes", "-p", required=True, type=list_of_strings, help="Phenotype(s) to perform the sweep (format example: ep_res,de_res,size_res)")
    parser.add_argument("--train_function", "-f", required=True, type=str, help="Name of the file (without file extention) to use to create the training function (should be found in configs/train_function_config)")

    args = parser.parse_args()

    with initialize(version_base=None, config_path="Configs"):
        cfg = compose(
            config_name="default",
            overrides=[f"model_config={args.model}", f"train_function_config={args.train_function}"],
        )

    if cfg.train_function_config.log_wandb:
        wandb.init(
            project = "TFE",
            config = get_clean_config(cfg),
            name = args.wandb_run_name,
            tags = ["debug"],
        )
        wandb.config["input_encoding"] = args.encoding
    
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
        indexes = np.where(mi > cfg.model_config.mutual_info_threshold)[0]
        print(f"Nb of selected features: {len(indexes)}")

        if args.encoding == "categorical":
            # - 1 is used to shoft the [0,1,2] range to the [-1,0,1] used in the paper  
            train_dataset = SNPResidualDataset(X_train[indexes].to_numpy(dtype=np.float32) - 1, y_train.to_numpy(dtype=np.float32))
            validation_dataset = SNPResidualDataset(X_val[indexes].to_numpy(dtype=np.float32) - 1, y_val.to_numpy(dtype=np.float32))
            embedding_type = EmbeddingType.Linear
            embedding_weight = None
        elif args.encoding == "frequency":
            train_dataset = SNPResidualDataset(convert_categorical_to_frequency(X_train[indexes].to_numpy()), y_train.to_numpy(dtype=np.float32))
            validation_dataset = SNPResidualDataset(convert_categorical_to_frequency(X_val[indexes].to_numpy()), y_val.to_numpy(dtype=np.float32))
            embedding_type = EmbeddingType.Linear
            embedding_weight = None
        elif args.encoding == "one_hot":
            train_dataset = SNPResidualDataset(X_train[indexes].to_numpy(dtype=np.int32), y_train.to_numpy(dtype=np.float32))
            validation_dataset = SNPResidualDataset(X_val[indexes].to_numpy(dtype=np.int32), y_val.to_numpy(dtype=np.float32))
            embedding_type = EmbeddingType.EmbeddingTable
            embedding_weight = torch.eye(3)
        elif args.encoding == "learned":
            train_dataset = SNPResidualDataset(X_train[indexes].to_numpy(dtype=np.int32), y_train.to_numpy(dtype=np.float32))
            validation_dataset = SNPResidualDataset(X_val[indexes].to_numpy(dtype=np.int32), y_val.to_numpy(dtype=np.float32))
            embedding_type = EmbeddingType.EmbeddingTable
            embedding_weight = None
        train_from_config(
            phenotype,
            cfg,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            model = instantiate(
                cfg.model_config.model,
                n_features = len(indexes),
                embedding_type = embedding_type,
                embedding_table_weight = embedding_weight
            ),
        )
    
if __name__ == "__main__":
    main()