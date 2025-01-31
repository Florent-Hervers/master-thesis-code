import wandb

if __name__ == "__main__":
    api = wandb.Api()
    run = api.run("flo230702/TFE/jo83oru3")
    for phenotype in ["ep_res", "de_res", "FESSEp_res", "FESSEa_res", "size_res", "MUSC_res"]:
        run.summary[f"correlation {phenotype}.max"] = max(run.history()[f"correlation {phenotype}"].dropna())
    run.summary.update()