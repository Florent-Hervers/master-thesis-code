import wandb

from utils import list_of_strings, train_from_config, get_clean_config
from argparse import ArgumentParser
from hydra import initialize, compose

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--model", "-m", required=True, type=str, help="Name of the file (without file extention) to define the model to train (should be found in configs/model_config)")
    parser.add_argument("--data", "-d", required=True, type=str, help="Name of the file (without file extention) to use for the data (should be found in configs/data)")
    parser.add_argument("--phenotypes", "-p", required=True, type=list_of_strings, help="Phenotype(s) to perform the sweep (format example: ep_res,de_res,size_res)")
    parser.add_argument("--wandb_run_name", "-w", required=False, type=str, help="String to use for the wandb run name")
    parser.add_argument("--train_function", "-f", required=True, type=str, help="Name of the file (without file extention) to use to create the training function (should be found in configs/train_function_config)")

    args = parser.parse_args()

    with initialize(version_base=None, config_path="Configs"):
        run_cfg = compose(
            config_name="default",
            overrides=[f"model_config={args.model}", f"data={args.data}", f"train_function_config={args.train_function}"]
        )

    wandb_config = get_clean_config(run_cfg)

    if run_cfg.train_function_config.log_wandb:
        wandb.init(
            name = args.wandb_run_name,
            project="TFE",
            config = wandb_config,
            tags = ["debug"]
        )

    for phenotype in args.phenotypes:
        train_from_config(phenotype, run_cfg)
