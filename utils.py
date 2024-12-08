import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def results_heatmap(df1:      pd.DataFrame,
                    df2:      pd.DataFrame, 
                    suptitle: str,
                    title1:   str,
                    title2:   str,
                    x_label:  str,
                    y_label:  str):
    """Display the heatmap of the two given dataframes. They should have the
    same index and columns values for a appropriate comparaison.

    Args:
        df1 (pd.DataFrame): first dataframe to display
        df2 (pd.DataFrame): second dataframe to display
        suptitle (str): title of the whole plot
        title1 (str): title for the first dataframe heatmap
        title2 (str): title for the second dataframe heatmap
        x_label (str): label for the x-axis (ie label of the index of the dataframes)
        y_label (str): label for the y-axis (ie label of the columns of the dataframes)
    """
    
    subtitle_font =  {"size": 14}
    title_font = {"weight": "bold" ,"size": 18}
    
    fig,(ax1, ax2) = plt.subplots(1,2, figsize=(17 , 6.5))

    fig.suptitle(suptitle, font=title_font)
    sns.heatmap(df1, annot= True, cmap = "YlGnBu", fmt= ".3f", linecolor="black", linewidths=0.5, ax= ax1)
    plt.yticks(rotation=0)
    ax1.set_title(title1, font=subtitle_font)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.tick_params(rotation=0)

    sns.heatmap(df2, annot = True, cmap = "YlGnBu", fmt= ".3f", linecolor="black", linewidths=0.5, ax=ax2)
    ax2.set_title(title2, font=subtitle_font)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    ax2.tick_params(rotation=0)
    plt.show()

def format_batch(dict: dict):
    """ Convert a dictonary containing x keys, each one containing a 1-D tensor of length y to a tensor of shape (y,x).
    This function is intended to be used to format SNPmarkersDataset batches created by a torch.utils.data.dataloader to be directly usable
    for a model when several phenotypes are set with the `set_phenotypes` proprety.

    Args:
        dict (dict): dictionary(containing x keys, each one containing a 1-D tensor of length y) to transform to tensor. 

    Returns:
        torch.Tensor: the resulting tensor (of shape (y,x)).
    """
    return torch.stack([dict[key] for key in dict.keys()], dim= 1)


import numpy as np
from scipy.stats import pearsonr
from torch.nn import Module, L1Loss
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import wandb
from typing import Union

def train_DL_model(
        model: Module,
        optimizer : Optimizer,
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        n_epoch: int,
        criterion = L1Loss(),
        scheduler: Union[None, LRScheduler] = None,
        phenotype: Union[str, None] = None,
        log_wandb: bool = True,
    ):
    """Define a basic universal training function that support wandb logging. Evaluation on the validation dataset is performed every epoch.

        Args:
            model (Module): The model to train.
            optimizer (Optimizer): Optimizer used to train the given model.
            train_dataloader (DataLoader): Dataloader containing the training dataset.
            validation_dataloader (DataLoader): Dataloader containing the training dataset.
            n_epoch (int): number of epoch to train the model
            criterion (_type_, optional): Function to use as loss function. Defaults to L1Loss().
            scheduler (None | LRScheduler, optional): LR scheduler object to perform lr Scheduling. Defaults to None.
            phenotype (str | None, optional): String containing the current phenotype studied. This is only used for logging. Defaults to None.
            log_wandb (bool, optional): If true, the loss and correlation are logged into wandb, otherwise they are printed after every epoch. Defaults to True.
        
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()

    print(f"Model architecture : \n {model}")
    print(f"Numbers of parameters: {sum(p.numel() for p in model.parameters())}")
    model.to(device)

    for epoch in range(n_epoch):
        train_loss = []
        model.train()
        for x,y in train_dataloader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            y = y.view(-1,1)
            loss = criterion(output, y)
            train_loss.append(loss.cpu().detach())
            loss.backward()
            optimizer.step()
        
        if log_wandb:
            wandb.log({
                    "epoch": epoch, 
                    f"train_loss {f'{phenotype}'  if phenotype is not None else ''}": np.array(train_loss).mean()
                }
            )

        print(f"Finished training for epoch {epoch}{f' for {phenotype}' if phenotype is not None else ''}.",
              f"{f'Train loss: {np.array(train_loss).mean()} ' if not log_wandb else ''}")

        val_loss = []
        predicted = []
        target = []
        with torch.no_grad():
            model.eval()
            for x,y in validation_dataloader:
                x,y = x.to(device), y.to(device)
                output = model(x)
                y = y.view(-1,1)
                loss = criterion(output, y)
                val_loss.append(loss.cpu().detach())
                if len(predicted) == 0:
                    predicted = output.cpu().detach()
                    target = y.cpu().detach()
                else:
                    predicted = np.concatenate((predicted, output.cpu().detach()), axis = 0)
                    target = np.concatenate((target, y.cpu().detach()), axis = 0)
        
            # Resize the vectors to be accepted in the pearsonr function
            predicted = predicted.reshape((predicted.shape[0],))
            target = target.reshape((target.shape[0],))
            
            if scheduler is not None:
                scheduler.step()
            
            if log_wandb:
                wandb.log({
                        "epoch": epoch, 
                        f"validation_loss {f'{phenotype}' if phenotype is not None else ''}": np.array(val_loss).mean(),
                        f"correlation {f'{phenotype}' if phenotype is not None else ''}": pearsonr(predicted, target).statistic,
                    }
                )
        print(f"Validation step for epoch {epoch}{f' for {phenotype}' if phenotype is not None else ''} finished!",
            f"{f' Correlation: {pearsonr(predicted, target).statistic}. Validation loss: {np.array(val_loss).mean()}' if not log_wandb else ''}")