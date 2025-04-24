import os
import yaml
import wandb

from argparse import BooleanOptionalAction
from hydra import initialize, compose
from functools import partial
from utils import train_from_config, get_clean_config, get_default_config_parser
from omegaconf import DictConfig
from train_residual import train_on_residuals

def update_config_and_train(phenotype: str, run_cfg: DictConfig, training_function = train_from_config):
    """Wrapper launching the wandb run and that update the config file with the parameters chosen by the sweep agent.

    Args:
        phenotype (str): phenotype on which the model should be trained on (should be a key of SNPmarkersDataset.phenotypes).
        run_cfg (DictConfig): hydra config file fetched using the compose API.
        training_function(function): the function to run at the end 
    """

    run = wandb.init(
        tags = ["tuning"]
    )

    # Update values in config based on what's chosen by the agent
    for k in run_cfg["model_config"].keys():
        if k in wandb.config.keys():
            run_cfg["model_config"][k] = wandb.config[k]

    run.config.update(get_clean_config(run_cfg))
    training_function(phenotype, run_cfg)

if __name__ == "__main__":
    parser = get_default_config_parser()
    parser.add_argument("--residual", default=False, action=BooleanOptionalAction, help="If True, perform the sweep on the residuals. Defaults to False")

    args = parser.parse_args()
    
    with open(os.path.join("Configs/sweeps",args.sweep_config + ".yaml"), "r") as f:
        sweep_cfg = yaml.safe_load(f)
    
    if args.all:
        # wrap the phenotypes in a list such that the following loop makes only one iteration
        args.phenotypes = [args.phenotypes]

    for i,phenotype in enumerate(args.phenotypes):

        if not args.all:
            # As the config isn't reset, we have to keep track of the previous phenotype for an appropriate update
            str_to_replace = "??"
            if i > 0:
                str_to_replace = args.phenotypes[i-1]
            sweep_cfg["name"] = " ".join([w.replace(str_to_replace, phenotype) for w in sweep_cfg["name"].split()])
            sweep_cfg["metric"]["name"] = " ".join([w.replace(str_to_replace, phenotype) for w in sweep_cfg["metric"]["name"].split()])
        else:
            str_to_replace = "??"
            # Use the first phenotype as the phenotype to optimize (only relevant in sweep using bayeasian opt)
            sweep_cfg["name"] = " ".join([w.replace(str_to_replace, phenotype[0]) for w in sweep_cfg["name"].split()])
            sweep_cfg["metric"]["name"] = " ".join([w.replace(str_to_replace, phenotype[0]) for w in sweep_cfg["metric"]["name"].split()])


        sweep_id = wandb.sweep(sweep=sweep_cfg, project="TFE")
        with initialize(version_base=None, config_path="Configs"):
            run_cfg = compose(
                config_name="default",
                overrides=[f"model_config={args.model}", f"data={args.data}", f"train_function_config={args.train_function}"]
            )
        
        if args.residual:
            wandb.agent(sweep_id, function=partial(update_config_and_train, phenotype, run_cfg, train_on_residuals))
        else:
            wandb.agent(sweep_id, function=partial(update_config_and_train, phenotype, run_cfg))