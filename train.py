import wandb

from utils import get_default_config_parser, train_from_config, get_clean_config
from hydra import initialize, compose

if __name__ == "__main__":
    parser = get_default_config_parser()
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

    if args.all:
        train_from_config(args.phenotypes, run_cfg)
    else:
        for phenotype in args.phenotypes:
            train_from_config(phenotype, run_cfg)
