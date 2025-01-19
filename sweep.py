from omegaconf import OmegaConf
from hydra import initialize, compose

from argparse import ArgumentParser
import yaml
import wandb
from functools import partial
from hydra.utils import instantiate
from omegaconf import DictConfig
from hydra.utils import call, instantiate


def train_from_config(phenotype: str, run_cfg: DictConfig, wandb_cfg: dict):
    if run_cfg.data.train_dataset._target_ == "dataset.SNPmarkersDataset":
        wandb.init(
            config = wandb_cfg,
            tags = ["tuning"]
        )

        # Update values in config based on what's chosen by the agent
        for k in OmegaConf.to_container(run_cfg.template.config, resolve=True).keys():
            run_cfg.template.config[k] = wandb.config[k]

        train_dataset= instantiate(run_cfg.data.train_dataset)
        train_dataset.set_phenotypes = phenotype
        validation_dataset = instantiate(run_cfg.data.validation_dataset)
        validation_dataset.set_phenotypes = phenotype
        call(run_cfg.template.train_function, 
            train_dataloader = instantiate(run_cfg.data.train_dataloader, dataset=train_dataset),
            validation_dataloader = instantiate(run_cfg.data.validation_dataloader, dataset=validation_dataset))
    else:
        raise Exception(f"Dataset template {run_cfg.data.train_dataset._target_} not supported yet")

def list_of_strings(arg):
    return arg.split(',')

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--sweep_config", "-s", required=True, type=str, help="Path to the YAML file containing the config file for the sweep")
    parser.add_argument("--template", "-t", required=True, type=str, help="Name of the file (without file extention) to use for the training (should be found in configs/template)")
    parser.add_argument("--data", "-d", required=True, type=str, help="Name of the file (without file extention) to use for the data (should be found in configs/data)")
    parser.add_argument("--phenotypes", "-p", required=True, type=list_of_strings, help="Phenotype(s) to perform the sweep (format example: ep_res,de_res,size_res)")
    

    args = parser.parse_args()
    
    with open(args.sweep_config, "r") as f:
        sweep_cfg = yaml.safe_load(f)
    
    for i,phenotype in enumerate(args.phenotypes):
        
        # As the config isn't reset, we have to keep track of the previous phenotype for an appropriate update
        str_to_replace = "??"
        if i > 0:
            str_to_replace = args.phenotypes[i-1]
        sweep_cfg["name"] = " ".join([w.replace(str_to_replace, phenotype) for w in sweep_cfg["name"].split()])
        sweep_cfg["metric"]["name"] = " ".join([w.replace(str_to_replace, phenotype) for w in sweep_cfg["metric"]["name"].split()])
        
        sweep_id = wandb.sweep(sweep=sweep_cfg, project="TFE")
        with initialize(version_base=None, config_path="configs"):
            run_cfg = compose(
                config_name="default",
                overrides=[f"template={args.template}", f"data={args.data}"]
            )
        
        dict_cfg = OmegaConf.to_container(run_cfg, resolve=True)
        # Contain a clean version of the config that only keep relevant info
        wandb_cfg = {}

        # Remove useless data and process the data from some keys that has interesting info for the logging
        dict_cfg["data"]["train_dataset"].pop("skip_check")
        dict_cfg["data"]["train_dataset"]["dataset"] = dict_cfg["data"]["train_dataset"].pop("_target_").split(".")[-1]
        wandb_cfg["train_dataset"] = dict_cfg["data"]["train_dataset"]
        
        dict_cfg["data"]["validation_dataset"].pop("skip_check")
        dict_cfg["data"]["validation_dataset"]["dataset"] = dict_cfg["data"]["validation_dataset"].pop("_target_").split(".")[-1]
        wandb_cfg["validation_dataset"] = dict_cfg["data"]["validation_dataset"]
        
        dict_cfg["template"]["train_function"]["model"].pop("dropout")
        dict_cfg["template"]["train_function"]["model"]["architecture"] = dict_cfg["template"]["train_function"]["model"].pop("_target_").split(".")[-1]

        for k,v in dict_cfg["template"]["train_function"]["model"].items():
            wandb_cfg[k] = v

        dict_cfg["template"]["train_function"]["optimizer"].pop("_target_")
        dict_cfg["template"]["train_function"]["optimizer"]["optimizer"] = dict_cfg["template"]["train_function"]["optimizer"].pop("optimizer_name")
        for k,v in dict_cfg["template"]["train_function"]["optimizer"].items():
            wandb_cfg[k] = v
        
        wandb_cfg["criterion"] = dict_cfg["template"]["train_function"]["criterion"].pop("_target_").split()[-1]

        # Remove the previously preprocessed keys
        dict_cfg["template"]["train_function"].pop("model")
        dict_cfg["template"]["train_function"].pop("optimizer")
        dict_cfg["template"]["train_function"].pop("criterion")

        # Remove useless keys before the merging
        dict_cfg["template"]["train_function"].pop("log_wandb")
        dict_cfg["template"]["train_function"].pop("train_dataloader")
        dict_cfg["template"]["train_function"].pop("validation_dataloader")
        dict_cfg["template"]["train_function"].pop("_target_")

        for k,v in dict_cfg["template"]["train_function"].items():
            wandb_cfg[k] = v
        
        wandb.agent(sweep_id, function=partial(train_from_config, phenotype, run_cfg, wandb_cfg))