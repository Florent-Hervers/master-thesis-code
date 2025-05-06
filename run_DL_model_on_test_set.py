
from sklearn.linear_model import Ridge
import torch
import numpy as np
import pandas as pd

from os import listdir
from hydra import compose, initialize
from hydra.utils import instantiate
from argparse import ArgumentParser
from Models.GPTransformer import EmbeddingType
from utils import convert_categorical_to_frequency, format_batch
from os.path import isfile, join
from scipy.stats import pearsonr
from dataset import SNPmarkersDataset, SNPResidualDataset
from torch.utils.data import DataLoader
from sklearn.feature_selection import mutual_info_regression
from functools import partial

def evaluate_models(checkpoint_directory: str):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    non_processed_files = []

    for model_file in [f for f in listdir(checkpoint_directory) if isfile(join(checkpoint_directory, f))]:
        model_filename = model_file.split(".")[0]
        model_name = model_filename.split("_")[0]
        if model_filename.split("_")[-1] == "res":
            phenotype = model_filename.split("_")[-2] + "_" + model_filename.split("_")[-1]
        elif model_filename.split("_")[-1] == "all":
            phenotype = ["ep_res", "de_res", "FESSEp_res", "FESSEa_res"]
        else:
            print(f"WARNING: Bad structured filename for {model_file}")
            continue
            
        
        # Only to test that the script works as intended
        if (model_filename != "GPTransformer_size_res" and model_name == "GPTransformer") or phenotype != "ep_res":
            continue
        
        print(f"///////////////////////////////// {model_filename} /////////////////////////////////")
        
        try:            
            if model_name == "GPTransformer" and phenotype != "size_res":
                mi = np.zeros(36304)
                modes = ["train", "validation", "test"]

                for mode in modes:
                    dataset = SNPmarkersDataset(mode = mode, skip_check=True)
                    dataset.set_phenotypes = phenotype
                    
                    X = dataset.get_all_SNP()
                    y = dataset.phenotypes[phenotype]

                    if mode == "validation":
                        X_val = X
                        y_val = y
                    elif mode == "test":
                        X_test = X
                        y_test = y
    
                    mi += mutual_info_regression(X,y, n_jobs=-1, discrete_features=True, random_state=2307)

                # Divide the number of modes to obtain the average mutual information
                mi /= len(modes)
                indexes = np.where(mi > 0.02)[0]
                
                if phenotype in  ["de_res", "FESSEp_Res", "FESSEa_res", "MUSC_res"] :
                    # - 1 is used to shoft the [0,1,2] range to the [-1,0,1] used in the paper  
                    test_dataset = SNPResidualDataset(X_test[indexes].to_numpy(dtype=np.float32) - 1, y_test.to_numpy(dtype=np.float32))
                    validation_dataset = SNPResidualDataset(X_val[indexes].to_numpy(dtype=np.float32) - 1, y_val.to_numpy(dtype=np.float32))
                    n_features = 0
                    embedding_type = EmbeddingType.Linear
                elif phenotype  == "ep_res":
                    test_dataset = SNPResidualDataset(convert_categorical_to_frequency(X_test[indexes].to_numpy()), y_test.to_numpy(dtype=np.float32))
                    validation_dataset = SNPResidualDataset(convert_categorical_to_frequency(X_val[indexes].to_numpy()), y_val.to_numpy(dtype=np.float32))
                    n_features = 0
                    embedding_type = EmbeddingType.Linear
                elif phenotype == "size_res":
                    test_dataset = SNPResidualDataset(X_test[indexes].to_numpy(dtype=np.int32), y_test.to_numpy(dtype=np.float32))
                    validation_dataset = SNPResidualDataset(X_val[indexes].to_numpy(dtype=np.int32), y_val.to_numpy(dtype=np.float32))
                    n_features = 3
                    embedding_type = EmbeddingType.EmbeddingTable
                else:
                    raise Exception(f"Phenotype {phenotype} isn't recognized!")  
                
                sequence_length = len(indexes) 

            elif (model_name == "GPTransformer" and phenotype == "size_res") or model_name == "GPTransformer2":
                original_train_dataset = SNPmarkersDataset(mode = "validation")
                original_validation_dataset = SNPmarkersDataset(mode = "test")

                original_train_dataset.set_phenotypes = phenotype
                original_validation_dataset.set_phenotypes = phenotype

                if model_name == "GPTransformer" and phenotype == "size_res":
                    token_file = "../Data/tokenized_genotype_5_8.csv" 
                elif model_name == "GPTransformer2" and model_filename.split("_")[1] == "4mer":
                    token_file = "../Data/tokenized_genotype_5_4.csv"
                elif model_name == "GPTransformer2" and model_filename.split("_")[1] == "aug":
                    token_file = "../Data/tokenized_genotype_9_4.csv"

                all_sequences_tokenized = pd.read_csv(token_file, index_col = 0)

                X_test = all_sequences_tokenized.loc[original_train_dataset.get_all_SNP().index]
                X_val = all_sequences_tokenized.loc[original_validation_dataset.get_all_SNP().index]
                
                y_test = original_train_dataset.phenotypes[phenotype]
                y_val = original_validation_dataset.phenotypes[phenotype]
                
                test_dataset = SNPResidualDataset(X_test.to_numpy(dtype=np.int32), y_test.to_numpy(dtype=np.float32))
                validation_dataset = SNPResidualDataset(X_val.to_numpy(dtype=np.int32), y_val.to_numpy(dtype=np.float32))
                
                TOKEN_SIZE = int(token_file.replace(".", "_").split("_")[-2])
                VOCAB_SIZE = int(token_file.replace(".", "_").split("_")[-3])
                
                n_features = VOCAB_SIZE**TOKEN_SIZE + 1
                sequence_length = len(X_test.iloc[0])
                embedding_type = EmbeddingType.EmbeddingTable

            elif model_name == "ReGS":
                original_train_dataset = SNPmarkersDataset(mode = "train")
                original_test_dataset = SNPmarkersDataset(mode = "test")
                original_validation_dataset = SNPmarkersDataset(mode = "validation")

                # Best ridge hyperparameters for all the phenotypes
                hp = {
                    "ep_res": {"lambda": 55600},
                    "de_res": {"lambda": 44500},
                    "FESSEp_res": {"lambda": 26250},
                    "FESSEa_res": {"lambda": 34000},
                    "size_res": {"lambda": 20900},
                    "MUSC_res": {"lambda": 23950},
                }
                
                original_test_dataset.set_phenotypes = phenotype
                original_validation_dataset.set_phenotypes = phenotype

                X_train = original_train_dataset.get_all_SNP()
                y_train = original_train_dataset.phenotypes[phenotype]

                X_val = original_validation_dataset.get_all_SNP()
                y_val = original_validation_dataset.phenotypes[phenotype]
                
                ridge_model = Ridge(alpha= hp[phenotype]["lambda"])
                    
                ridge_model.fit(X_train, y_train)
               
                y_test_pre = ridge_model.predict(X_test)
                y_val_pre = ridge_model.predict(X_val)

                y_test = y_test - y_test_pre
                y_val = y_val - y_val_pre

                y_test = y_test.to_numpy(dtype=np.float32)
                y_val = y_val.to_numpy(dtype=np.float32)
                
                test_dataset = SNPResidualDataset(X_test.to_numpy(dtype=np.float32), y_test)
                validation_dataset = SNPResidualDataset(X_val.to_numpy(dtype=np.float32), y_val)

            else:
                validation_dataset = SNPmarkersDataset(mode = "validation")
                test_dataset = SNPmarkersDataset(mode = "test")

                validation_dataset.set_phenotypes = phenotype
                test_dataset.set_phenotypes = phenotype

            if model_filename.split("_")[-1] == "all":
                validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False, collate_fn=partial(format_batch, phenotypes=phenotype))
                test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=partial(format_batch, phenotypes=phenotype))
            else:
                validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
                test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            
        except Exception as e:
            print(f"The following error occured when preprocessing data for the model for {model_filename}: \n{e}")
            non_processed_files.append(model_filename)
            continue
        
        try:
            if model_name == "GPTransformer2":
                config_filename = f"Test_GPTransformer2_{model_filename.split('_')[1]}"
            else:
                config_filename = f"Test_{model_filename}"

            with initialize(version_base=None, config_path="Configs"):
                model_cfg = compose(
                    config_name="default_test",
                    overrides=[f"model_config={config_filename}"]
                )

            if model_name == "GPTransformer" or model_name == "GPTransformer2":
                model = instantiate(
                    model_cfg.model_config.model,
                    n_features = n_features,
                    sequence_length = sequence_length,
                    embedding_type = embedding_type
                )
            else:
                model = instantiate(
                    model_cfg.model_config.model,
                    output_size = len(phenotype) if type(phenotype) == list else 1
                )

            model.load_state_dict(torch.load(join(checkpoint_directory, model_file,), weights_only=False))
            model.eval()

        except Exception as e:
            print(f"The following error occured when loading the model for {model_filename}: \n{e}")
            non_processed_files.append(model_filename)
            continue
        
        try:
            for set_name, dataloader in zip(["validation", "test"], [validation_dataloader, test_dataloader]):
                test_MAE = []
                predicted = []
                target = []
                with torch.no_grad():
                    for x,y in dataloader:
                        x,y = x.to(device), y.to(device)
                        output = model(x)
                        if len(y.shape) == 1:
                            y = y.view(-1,1)

                        loss = torch.nn.functional.l1_loss(output, y)
                        test_MAE.append(loss.cpu().detach())
                        if len(predicted) == 0:
                            predicted = output.cpu().detach()
                            target = y.cpu().detach()
                        else:
                            predicted = np.concatenate((predicted, output.cpu().detach()), axis = 0)
                            target = np.concatenate((target, y.cpu().detach()), axis = 0)
                    
                    # Resize the vectors to be accepted in the pearsonr function
                    predicted = predicted.reshape((predicted.shape[0],) if type(phenotype) == str else (predicted.shape[0],len(phenotype))) 
                    target = target.reshape((target.shape[0],) if type(phenotype) == str else (target.shape[0],len(phenotype)))
                    
                    if model_name == "ResGS":
                        if set_name == "validation":
                            predicted += y_val_pre
                            target += y_val_pre
                        elif set_name == "test":
                            predicted += y_test_pre
                            target += y_test_pre

                    print(f"{model_name} results on the {set_name} set for the phenotype {phenotype}:")
                    print(f"    - MAE: {np.array(test_MAE).mean()}")
                    print(f"    - Correlation: {pearsonr(predicted, target).statistic}")
        
        except Exception as e:
            print(f"The following error occured when running the model for {model_filename}: \n{e}")
            non_processed_files.append(model_filename)
            continue

    if len(non_processed_files) != 0:
        print(f"WARNING : The following files wasn't processed due to an error: {non_processed_files}")
    else:
        print("Evaluation finished sucessfully.")

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-p", "--path", type=str, required=False, default="Trained_models", help="Path to the directory where all the models are stored")
    evaluate_models(parser.parse_args().path)