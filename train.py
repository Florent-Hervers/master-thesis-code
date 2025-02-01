import wandb

from utils import list_of_strings, train_from_config, get_clean_config
from argparse import ArgumentParser
from hydra import initialize, compose

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--template", "-t", required=True, type=str, help="Name of the file (without file extention) to use for the training (should be found in configs/template)")
    parser.add_argument("--data", "-d", required=True, type=str, help="Name of the file (without file extention) to use for the data (should be found in configs/data)")
    parser.add_argument("--phenotypes", "-p", required=True, type=list_of_strings, help="Phenotype(s) to perform the sweep (format example: ep_res,de_res,size_res)")
    parser.add_argument("--wandb_run_name", "-w", required=False, type=str, help="String to use for the wandb run name")

    args = parser.parse_args()

    with initialize(version_base=None, config_path="Configs"):
        run_cfg = compose(
            config_name="default",
            overrides=[f"template={args.template}", f"data={args.data}"]
        )

    if run_cfg.template.train_function.log_wandb:
        wandb.init(
            name = args.wandb_run_name,
            project="TFE",
            config = get_clean_config(run_cfg),
            tags = ["debug"]
        )

    for phenotype in args.phenotypes:
        train_from_config(phenotype, run_cfg)
