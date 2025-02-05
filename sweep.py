import os
import yaml
import wandb

from argparse import ArgumentParser
from hydra import initialize, compose
from functools import partial
from utils import train_from_config, list_of_strings, get_clean_config
from omegaconf import DictConfig

def update_config_and_train(phenotype: str, run_cfg: DictConfig):
    """Wrapper launching the wandb run and that update the config file with the parameters chosen by the sweep agent.

    Args:
        phenotype (str): phenotype on which the model should be trained on (should be a key of SNPmarkersDataset.phenotypes).
        run_cfg (DictConfig): hydra config file fetched using the compose API.
    """
    run = wandb.init(
        tags = ["tuning"]
    )

    # Update values in config based on what's chosen by the agent
    for k in run_cfg["model_config"].keys():
        if k in wandb.config.keys():
            run_cfg["model_config"][k] = wandb.config[k]

    run.config.update(get_clean_config(run_cfg))

    train_from_config(phenotype, run_cfg)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--sweep_config", "-s", required=True, type=str, help="Name of the file (without file extention) to use for the sweep configuration (should be found in configs/sweeps)")
    parser.add_argument("--model", "-m", required=True, type=str, help="Name of the file (without file extention) to create the model to train (should be found in configs/model_config)")
    parser.add_argument("--data", "-d", required=True, type=str, help="Name of the file (without file extention) to use for the data (should be found in configs/data)")
    parser.add_argument("--phenotypes", "-p", required=True, type=list_of_strings, help="Phenotype(s) to perform the sweep (format example: ep_res,de_res,size_res)")
    parser.add_argument("--train_function", "-f", required=True, type=str, help="Name of the file (without file extention) to use to create the training function (should be found in configs/train_function_config)")

    args = parser.parse_args()
    
    with open(os.path.join("Configs/sweeps",args.sweep_config + ".yaml"), "r") as f:
        sweep_cfg = yaml.safe_load(f)
    
    for i,phenotype in enumerate(args.phenotypes):

        # As the config isn't reset, we have to keep track of the previous phenotype for an appropriate update
        str_to_replace = "??"
        if i > 0:
            str_to_replace = args.phenotypes[i-1]
        sweep_cfg["name"] = " ".join([w.replace(str_to_replace, phenotype) for w in sweep_cfg["name"].split()])
        sweep_cfg["metric"]["name"] = " ".join([w.replace(str_to_replace, phenotype) for w in sweep_cfg["metric"]["name"].split()])
        
        sweep_id = wandb.sweep(sweep=sweep_cfg, project="TFE")
        with initialize(version_base=None, config_path="Configs"):
            run_cfg = compose(
                config_name="default",
                overrides=[f"model_config={args.model}", f"data={args.data}", f"train_function_config={args.train_function}"]
            )
        
        wandb.agent(sweep_id, function=partial(update_config_and_train, phenotype, run_cfg))