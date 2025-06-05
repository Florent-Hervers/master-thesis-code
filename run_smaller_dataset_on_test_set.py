import torch
import numpy as np

from os import listdir
from omegaconf import OmegaConf
from hydra import compose, initialize
from hydra.utils import instantiate
from argparse import ArgumentParser
from os.path import isfile, join
from scipy.stats import pearsonr
from dataset import SNPmarkersDataset
from torch.utils.data import DataLoader

def evaluate_training_set_size(checkpoint_directory: str):
    """Evaluate on the validation and the test set the models trained on smaller training sets.
    The models should be trained to predict the ep_res phenotype and the filename should be structured like this:
    \<dataset\>_\<model_name\>.pth. The name of configuration file of the model should be Test_\<model_name\>_ep_res.
    In case of error, the error will be printed and the next model will be evaluated. 
    At the end, all models that failed will be indicated to help debugging.

    Args:
        checkpoint_directory (str): The path to the directory where the models are stored
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    non_processed_files = []
    
    # Only the ep_res phenotype was evaluated
    phenotype = "ep_res"

    for model_file in [f for f in listdir(checkpoint_directory) if isfile(join(checkpoint_directory, f))]:
        model_filename = model_file.split(".")[0]
        model_name = model_filename.split("_")[1]
        dataset = model_filename.split("_")[0]
        print(f"///////////////////////////////// {model_filename} /////////////////////////////////")
        
        try:            
            validation_dataset = SNPmarkersDataset(mode = "validation")
            test_dataset = SNPmarkersDataset(mode = "test")

            validation_dataset.set_phenotypes = phenotype
            test_dataset.set_phenotypes = phenotype

            validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
            test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            
        except Exception as e:
            print(f"The following error occured when preprocessing data for the model for {model_filename}: \n{e}")
            non_processed_files.append(model_filename)
            continue
        
        try:
            config_filename = f"Test_{model_name}_{phenotype}"

            # Define custom resolver for the computation of the hidden_nodes for Deep_MLP (See https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html)
            OmegaConf.register_new_resolver("mul", lambda x, y: int(x * y), replace= True)
            OmegaConf.register_new_resolver("div", lambda x, y: int(x / y), replace= True)
            
            with initialize(version_base=None, config_path="Configs"):
                model_cfg = compose(
                    config_name="default_test",
                    overrides=[f"model_config={config_filename}"]
                )

            model = instantiate(
                model_cfg.model_config.model,
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

                        if type(phenotype) == str:
                            loss = torch.nn.functional.l1_loss(output, y).cpu().detach()
                        else:
                            loss = []
                            for i in range(len(phenotype)): 
                                loss.append(torch.nn.functional.l1_loss(output[:,i], y[:,i]).cpu().detach())

                        test_MAE.append(loss)
                        if len(predicted) == 0:
                            predicted = output.cpu().detach()
                            target = y.cpu().detach()
                        else:
                            predicted = np.concatenate((predicted, output.cpu().detach()), axis = 0)
                            target = np.concatenate((target, y.cpu().detach()), axis = 0)
                    
                    # Resize the vectors to be accepted in the pearsonr function
                    predicted = predicted.reshape((predicted.shape[0],) if type(phenotype) == str else (predicted.shape[0],len(phenotype))) 
                    target = target.reshape((target.shape[0],) if type(phenotype) == str else (target.shape[0],len(phenotype)))
                    
                    if type(phenotype) == str:
                        correlation = pearsonr(predicted, target).statistic
                        MAE = np.array(test_MAE).mean()
                    else:
                        correlation = []
                        MAE = []
                        for i in range(len(phenotype)):
                            correlation.append(pearsonr(predicted[:,i], target[:,i]).statistic)
                            MAE.append(np.array(test_MAE)[:,i].mean())
    
                    print(f"{model_name} results trained on the {dataset} on the {set_name} set for the phenotype {phenotype}:")
                    print(f"    - MAE: {MAE}")
                    print(f"    - Correlation: {correlation}")
        
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

    parser.add_argument("-p", "--path", type=str, required=False, default="Dataset_models", help="Path to the directory where all the models trained on the smaller training set are stored")
    evaluate_training_set_size(parser.parse_args().path)