import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
import numpy as np
from scipy.stats import pearsonr, ConstantInputWarning
from torch.nn import Module, L1Loss
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import wandb
from typing import Union, List
from functools import partial
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import functools
from hydra.utils import call, instantiate


def results_heatmap(df1:      pd.DataFrame,
                    df2:      pd.DataFrame, 
                    suptitle: str,
                    title1:   str,
                    title2:   str,
                    x_label:  str,
                    y_label:  str,
                    vertical: bool = False):
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
    
    if not vertical:
        fig,(ax1, ax2) = plt.subplots(1,2, figsize=(17 , 6.5))
    else:
        fig,(ax1, ax2) = plt.subplots(2,1, figsize=(17 , 10))

    fig.suptitle(suptitle, font=title_font)
    sns.heatmap(df1, annot= True, cmap = "mako", fmt= ".3f", linecolor="black", linewidths=0.5, ax= ax1)
    plt.yticks(rotation=0)
    ax1.set_title(title1, font=subtitle_font)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    ax1.tick_params(rotation=0)

    sns.heatmap(df2, annot = True, cmap = "mako", fmt= ".3f", linecolor="black", linewidths=0.5, ax=ax2)
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

def train_DL_model(
        model: Module,
        optimizer : Union[Optimizer, partial],
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        n_epoch: int,
        criterion = L1Loss(),
        scheduler: Union[None, LRScheduler] = None,
        phenotype: Union[str, None] = None,
        log_wandb: bool = True,
        initial_phenotype = None,
        validation_mean: float = 0,
        validation_std: float = 1,
        early_stop_threshold: float = 0,
        early_stop_n_epoch: int = 10,
    ):
    """Define a basic universal training function that support wandb logging. Evaluation on the validation dataset is performed every epoch.

        Args:
            model (Module): The model to train.
            optimizer (Optimizer or functool.partial): Optimizer used to train the given model. \
            A partial optimizer (without the model parameters) can be used instead of the already instansitated object.
            train_dataloader (DataLoader): Dataloader containing the training dataset.
            validation_dataloader (DataLoader): Dataloader containing the training dataset.
            n_epoch (int): number of epoch to train the model
            criterion (_type_, optional): Function to use as loss function. Defaults to L1Loss().
            scheduler (None | LRScheduler, optional): LR scheduler object to perform lr Scheduling. Defaults to None.
            phenotype (str | None, optional): String containing the current phenotype studied. This is only used for logging. Defaults to None.
            log_wandb (bool, optional): If true, the loss and correlation are logged into wandb, otherwise they are printed after every epoch. Defaults to True.
            initial_phenotype (np.array | None, optionnal): If the model compute a residual phenotype, you can provide the basis to see the evolution of correlation of the final prediction. Defaults to None.
            validation_mean (float, optional): mean of the phenotypes from the validation set that was substratcted when normalizing the validation phenotype. \
            This value will be added back to the validation target and the model prediction on the validation set to have comparable validation loss. Defaults to 0.
            validation_std (float, optional): standard deviation of the phenotypes from the validation set that was used as denominator when normalizing the validation phenotype. \
            This value will be added back to the validation target and the model prediction on the validation set to have comparable validation loss. Defaults to 1.
            early_stop_threshold (float, optional): percentage of the maximum correlation below the early stop counter stop incrementing. This value should stay between 0 and 1 included. Defaults to 0.
            early_stop_n_epoch (int, optional): number of consecutive epoch where the correlation is below early_stop_threshold*max_correlation. Defaults to 10.
    """
    if early_stop_threshold > 1 and early_stop_threshold < 0:
        raise Exception("Early stop threshold should be between 0 and 1")
    
    if early_stop_n_epoch > n_epoch:
        raise Exception(f"Number of epoch before performing early stop '{early_stop_n_epoch})should be less that the max amount of epoch ({n_epoch})")

    if type(optimizer) == partial:
        try:
            optimizer = optimizer(model.parameters())
        except Exception as e:
            raise Exception(f"The following error occured when completing the optimizer: {e.args}")

        assert type(optimizer) != Optimizer, \
            "The partial optimizer given don't yield a Optimizer class when the model parameters are given"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    print(f"Device used: {device}")
    print(f"Model architecture : \n {model}")
    print(f"Numbers of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Optimizer used: {optimizer}")
    
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Train feature batch shape: {train_features.size()}")
    print(f"Train labels batch shape: {train_labels.size()}")
    
    val_features, val_labels = next(iter(train_dataloader))
    print(f"Validation feature batch shape: {val_features.size()}")
    print(f"Validation labels batch shape: {val_labels.size()}")

    model.to(device)

    max_correlation = 0
    early_stop_counter = 0

    # Define the keys for the dictonary used by wandb to avoid mispelling mistakes
    correlation_key = f"correlation {f'{phenotype}' if phenotype is not None else ''}"
    train_loss_key = f"train_loss {f'{phenotype}'  if phenotype is not None else ''}"
    epoch_key = "epoch"
    validation_loss_key = f"validation_loss {f'{phenotype}' if phenotype is not None else ''}"
    
    # Define metrics to improve the results display
    if log_wandb:
        wandb.define_metric(correlation_key, step_metric=epoch_key, summary='max')
        wandb.define_metric(train_loss_key, step_metric=epoch_key, summary='none')
        wandb.define_metric(validation_loss_key, step_metric=epoch_key, summary='none')
        wandb.define_metric(epoch_key, hidden=True, summary='none')

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
                    epoch_key: epoch, 
                    train_loss_key: np.array(train_loss).mean()
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
                
                # Scale back the values in order to have the validation loss on the unscaled values
                output = (output * validation_std) + validation_mean
                y = (y * validation_std) + validation_mean
                
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

            if initial_phenotype is not None:
                initial_phenotype = initial_phenotype.reshape((initial_phenotype.shape[0],))
                predicted = initial_phenotype + predicted
                target = initial_phenotype + target
            
            if scheduler is not None:
                scheduler.step()
            
            
        with warnings.catch_warnings(record=True) as w:
            correlation = pearsonr(predicted, target).statistic
            if len(w) > 0:
                print(f"Stop execution as model converged to outputing always the same value ({predicted[0]})")
                for warning in w:
                    print(warning)
                if log_wandb:
                    wandb.finish(1)
                
                return
            
        if log_wandb:
            wandb.log({validation_loss_key: np.array(val_loss).mean(), correlation_key: correlation,})
        
        print(f"Validation step for epoch {epoch}{f' for {phenotype}' if phenotype is not None else ''} finished!",
            f"{f' Correlation: {correlation}. Validation loss: {np.array(val_loss).mean()}' if not log_wandb else ''}")
        
        if correlation > max_correlation:
            max_correlation = correlation
        
        # Take the absolute value such that small oscillation arround zero doesn't reset the counter
        if abs(correlation) < early_stop_threshold * max_correlation:
            if early_stop_counter == 0:
                print(f"Start early stop counter, current threshold is {early_stop_threshold * max_correlation}")
            early_stop_counter +=1
        else:
            early_stop_counter = 0
        
        if early_stop_counter >= early_stop_n_epoch:
            print(f"Early stop condition met. Best correlation observed: {max_correlation}")
            break

def print_elapsed_time(start_time: float):
    """Returns elapsed time following the format d h m s

    Args:
        start_time (float): the start time taken with the `time.time` function

    Returns:
        str: the formatted time elapsed from the given time
    """
    days, rem = divmod(time.time() - start_time, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, secondes = divmod(rem, 60)
    return f"{int(days)}d {int(hours)}h {int(minutes)}m {int(secondes)}s"

def get_partial_optimizer(optimizer_name: str, **kwargs):
    """Dummy function use to produced a partial optimizer from a hydra config file. This solution enable to consider \
    the optimization class not as a string (due to hydra) but as the function to call

    Args:
        optimizer_name (str): Name of the optimizer to use (should be a class of torch.optim)

    Returns:
        functools.partial: a partial function waiting the model parameters to yield the torch optimizer.
    """
    return functools.partial(getattr(torch.optim, optimizer_name), **kwargs)

def train_from_config(phenotype: str, run_cfg: DictConfig):
    """Launch the training based on the parameters from the hydra config files.

    Args:
        phenotype (str): phenotype on which the model should be trained on (should be a key of SNPmarkersDataset.phenotypes).
        run_cfg (DictConfig): hydra config file fetch using the compose API.

    Raises:
        Exception: in case of unknown dataset class
    """
    if run_cfg.data.train_dataset._target_ == "dataset.SNPmarkersDataset":
        train_dataset= instantiate(run_cfg.data.train_dataset)
        train_dataset.set_phenotypes = phenotype
        validation_dataset = instantiate(run_cfg.data.validation_dataset)
        validation_dataset.set_phenotypes = phenotype

        # Define custom resolver for the computation of the hidden_nodes for Deep_MLP (See https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html)
        OmegaConf.register_new_resolver("mul", lambda x, y: int(x * y), replace= True)
        OmegaConf.register_new_resolver("div", lambda x, y: int(x / y), replace= True)

        call(run_cfg.template.train_function,
            phenotype = phenotype, 
            train_dataloader = instantiate(run_cfg.data.train_dataloader, dataset=train_dataset),
            validation_dataloader = instantiate(run_cfg.data.validation_dataloader, dataset=validation_dataset))
    else:
        raise Exception(f"Dataset template {run_cfg.data.train_dataset._target_} not supported yet")

def list_of_strings(arg):
    """Function defining a custom class for argument parsing."""
    return arg.split(',')

def get_clean_config(run_cfg: dict):
    """Create a clean dictionary can only contain useful info on the current run based on the hydra config.

    Args:
        run_cfg (dict): hydra config file fetch using the compose API.

    Returns:
        dict: clean dictionary usable as config for the wandb run.
    """
    
    # Define custom resolver for the computation of the hidden_nodes for Deep_MLP (See https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html)
    OmegaConf.register_new_resolver("mul", lambda x, y: int(x * y), replace= True)
    OmegaConf.register_new_resolver("div", lambda x, y: int(x / y), replace= True)

    dict_cfg = OmegaConf.to_container(run_cfg, resolve=True)
    # Contain a clean version of the config that only keep relevant info
    wandb_cfg = {}

    # Remove useless data and process the data from some keys that has interesting info for the logging
    wandb_cfg["batch_size"] = dict_cfg["template"]["config"]["batch_size"]
    
    dict_cfg["data"]["train_dataset"].pop("skip_check")
    dict_cfg["data"]["train_dataset"]["dataset"] = dict_cfg["data"]["train_dataset"].pop("_target_").split(".")[-1]
    wandb_cfg["train_dataset"] = dict_cfg["data"]["train_dataset"]
    
    dict_cfg["data"]["validation_dataset"].pop("skip_check")
    dict_cfg["data"]["validation_dataset"]["dataset"] = dict_cfg["data"]["validation_dataset"].pop("_target_").split(".")[-1]
    wandb_cfg["validation_dataset"] = dict_cfg["data"]["validation_dataset"]
    
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
    dict_cfg["template"]["train_function"].pop("phenotype")
    dict_cfg["template"]["train_function"].pop("_target_")

    for k,v in dict_cfg["template"]["train_function"].items():
        wandb_cfg[k] = v
    
    return wandb_cfg

def results_1_dimentions(array1:      list,
                         array2:      list, 
                         x_values:    list,
                         suptitle:    str,
                         title1:      str,
                         title2:      str,
                         x_label:     str,
                         y1_label:    str,
                         y2_label:    str,
                         vertical:    bool = False):
    """Display the evolution of two given metrics given a parameter.

    Args:
        array1 (list): first letric data array
        array2 (list): second metric data array
        x_values(list): parameter values used
        suptitle (str): title of the whole plot
        title1 (str): title for the first graph
        title2 (str): title for the second graph
        x_label (str): label for the x-axis
        y1_label (str): label for the y-axis of the first graph
        y2_label (str): label for the y-axis of the second graph 
    """
    
    subtitle_font =  {"size": 14}
    title_font = {"weight": "bold" ,"size": 18}
    
    if not vertical:
        fig,(ax1, ax2) = plt.subplots(1,2, figsize=(17 , 6.5))
    else:
        fig,(ax1, ax2) = plt.subplots(2,1, figsize=(17 , 10))

    fig.suptitle(suptitle, font=title_font)
    sns.scatterplot(x=x_values, y=array1, ax=ax1)
    plt.yticks(rotation=0)
    ax1.set_title(title1, font=subtitle_font)
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label)
    ax1.tick_params(rotation=0)

    sns.scatterplot(x=x_values, y=array2, ax=ax2)
    ax2.set_title(title2, font=subtitle_font)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y2_label)
    ax2.tick_params(rotation=0)
    plt.show()