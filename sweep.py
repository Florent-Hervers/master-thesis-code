import os
from omegaconf import OmegaConf
from hydra import initialize, compose

from argparse import ArgumentParser
import yaml
import wandb
from functools import partial
from utils import train_from_config, list_of_strings, get_clean_config
from omegaconf import DictConfig

def update_config_and_train(phenotype: str, run_cfg: DictConfig, wandb_cfg: dict):
    """Wrapper launching the wandb run and that update the config file with the parameters chosen by the sweep agent.

    Args:
        phenotype (str): phenotype on which the model should be trained on (should be a key of SNPmarkersDataset.phenotypes).
        run_cfg (DictConfig): hydra config file fetch using the compose API.
        wandb_cfg (dict): dictionary given to wandb as config.
    """
    wandb.init(
        config = wandb_cfg,
        tags = ["tuning"]
    )

    # Update values in config based on what's chosen by the agent
    for k in OmegaConf.to_container(run_cfg.template.config, resolve=True).keys():
        run_cfg.template.config[k] = wandb.config[k]

    train_from_config(phenotype, run_cfg)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--sweep_config", "-s", required=True, type=str, help="Name of the file (without file extention) to use for the sweep configuration (should be found in configs/sweeps)")
    parser.add_argument("--template", "-t", required=True, type=str, help="Name of the file (without file extention) to use for the training (should be found in configs/template)")
    parser.add_argument("--data", "-d", required=True, type=str, help="Name of the file (without file extention) to use for the data (should be found in configs/data)")
    parser.add_argument("--phenotypes", "-p", required=True, type=list_of_strings, help="Phenotype(s) to perform the sweep (format example: ep_res,de_res,size_res)")
    

    args = parser.parse_args()
    
    with open(os.path.join("configs/sweeps",args.sweep_config + ".yaml"), "r") as f:
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
        
        wandb_cfg = get_clean_config(run_cfg)
        
        wandb.agent(sweep_id, function=partial(update_config_and_train, phenotype, run_cfg, wandb_cfg))