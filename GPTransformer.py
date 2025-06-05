import torch
import pandas as pd
import warnings
import wandb
import numpy as np

from dataset import SNPmarkersDataset, SNPResidualDataset
from utils import train_from_config, convert_categorical_to_frequency, get_clean_config, get_default_config_parser
from sklearn.feature_selection import mutual_info_regression
from hydra import compose,initialize
from hydra.utils import instantiate
from Models.GPTransformer import EmbeddingType

def main():
    """Prepare the dataset depending on the arguments of the script and launch the training of the GPTransformer model.

    Raises:
        Exception: 
            if the selection argument is tokenization and the token filename doesn't 
            satisfy the pattern *_{VOCAB_SIZE}_{TOKEN_SIZE}.csv
    """
    parser = get_default_config_parser()

    parser.add_argument(
        "-e",
        "--encoding",
        choices=["categorical", "frequency", "one_hot", "learned"],
        required=True,
        help="Encoding of the input of the model to use (see paper for more detail)"
    )
    parser.add_argument(
        "--selection",
        "-s",
        required=True,
        choices = ["mutual_information", "tokenization"],
        help="Type of sequences reduction to apply the model"
    )

    parser.add_argument(
        "--tokenized_sequences_filepath",
        "-t",
        type=str,
        default="../Data/tokenized_genotype_5_8.csv",
        help="Path to the csv file containing the tokenized sequences. The token filename must satisfy the pattern *_TOKEN_SIZE.csv. Defaults to ../Data/tokenized_genotype_8.csv"
    )

    args = parser.parse_args()

    with initialize(version_base=None, config_path="Configs"):
        cfg = compose(
            config_name="default",
            overrides=[f"model_config={args.model}", f"data={args.data}", f"train_function_config={args.train_function}"],
        )

    if cfg.train_function_config.log_wandb:
        wandb.init(
            project = "TFE",
            config = get_clean_config(cfg),
            name = args.wandb_run_name,
            tags = ["debug"],
        )
        wandb.config["input_encoding"] = args.encoding
        wandb.config["features_processing"] = args.selection
    
    selected_phenotypes = args.phenotypes
    
    if args.all:
        selected_phenotypes = [selected_phenotypes]
    
    for phenotype in selected_phenotypes: 
        
        if args.selection == "mutual_information":
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
                if args.all:
                    phenotypes = []
                    for pheno in phenotype:
                        phenotypes.append(dataset.phenotypes[pheno].to_numpy())

                    # Convert to a Dataframe to reuse the dataset generation code without modification
                    y = pd.DataFrame(np.stack(phenotypes, axis = 1))
                else:
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
            elif args.encoding == "frequency":
                train_dataset = SNPResidualDataset(convert_categorical_to_frequency(X_train[indexes].to_numpy()), y_train.to_numpy(dtype=np.float32))
                validation_dataset = SNPResidualDataset(convert_categorical_to_frequency(X_val[indexes].to_numpy()), y_val.to_numpy(dtype=np.float32))
            else:
                train_dataset = SNPResidualDataset(X_train[indexes].to_numpy(dtype=np.int32), y_train.to_numpy(dtype=np.float32))
                validation_dataset = SNPResidualDataset(X_val[indexes].to_numpy(dtype=np.int32), y_val.to_numpy(dtype=np.float32))
            
            if args.encoding == "categorical" or args.encoding == "frequency":
                n_features = 0
            else:
                n_features = 3

            sequence_length = len(indexes)
            
            
        elif args.selection == "tokenization":

            if args.encoding != "learned":
                warnings.warn("The only encoding valid with tokenization is the learned encoding. Execution continues used the learned embedding")
                args.encoding = "learned"

            original_train_dataset = instantiate(cfg.data.train_dataset)
            original_validation_dataset = instantiate(cfg.data.validation_dataset)

            original_train_dataset.set_phenotypes = phenotype
            original_validation_dataset.set_phenotypes = phenotype

            all_sequences_tokenized = pd.read_csv(args.tokenized_sequences_filepath, index_col = 0)

            X_train = all_sequences_tokenized.loc[original_train_dataset.get_all_SNP().index]
            X_val = all_sequences_tokenized.loc[original_validation_dataset.get_all_SNP().index]
            
            if args.all:
                train_phenotypes = []
                val_phenotypes = []
                for pheno in phenotype:
                    train_phenotypes.append(original_train_dataset.phenotypes[pheno].to_numpy())
                    val_phenotypes.append(original_validation_dataset.phenotypes[pheno].to_numpy())
                
                # Convert to a Dataframe to reuse the dataset generation code without modification
                y_train = pd.DataFrame(np.stack(train_phenotypes, axis = 1))
                y_val = pd.DataFrame(np.stack(val_phenotypes, axis = 1))
            else:
                y_train = original_train_dataset.phenotypes[phenotype]
                y_val = original_validation_dataset.phenotypes[phenotype]
            
            train_dataset = SNPResidualDataset(X_train.to_numpy(dtype=np.int32), y_train.to_numpy(dtype=np.float32))
            validation_dataset = SNPResidualDataset(X_val.to_numpy(dtype=np.int32), y_val.to_numpy(dtype=np.float32))
            

            TOKEN_SIZE = args.tokenized_sequences_filepath.replace(".", "_").split("_")[-2]
            VOCAB_SIZE = args.tokenized_sequences_filepath.replace(".", "_").split("_")[-3]
            try:
                TOKEN_SIZE = int(TOKEN_SIZE)
                VOCAB_SIZE = int(VOCAB_SIZE)
            except:
                raise Exception("The token filename must satisfy the pattern *_{VOCAB_SIZE}_{TOKEN_SIZE}.csv")
            n_features = VOCAB_SIZE**TOKEN_SIZE + 1
            sequence_length = len(X_train.iloc[0])
        else:
            raise Exception("The argument {args.selection} isn't supported! This shouldn't happend if the ArgumentParser is correctly set!")   
            

        if args.encoding == "categorical":
            embedding_type = EmbeddingType.Linear
            embedding_weight = None
        elif args.encoding == "frequency":
            embedding_type = EmbeddingType.Linear
            embedding_weight = None
        elif args.encoding == "one_hot":
            embedding_type = EmbeddingType.EmbeddingTable
            embedding_weight = torch.eye(3)
        elif args.encoding == "learned":
            embedding_type = EmbeddingType.EmbeddingTable
            embedding_weight = None
        
        train_from_config(
            phenotype,
            cfg,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            model = instantiate(
                cfg.model_config.model,
                n_features = n_features,
                sequence_length = sequence_length,
                embedding_type = embedding_type,
                embedding_table_weight = embedding_weight,
                output_size = len(phenotype) if args.all else 1
            ),
            model_save_path = args.output_path
        )
    
if __name__ == "__main__":
    main()