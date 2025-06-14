import functools
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import time
import numpy as np
import wandb

from argparse import ArgumentParser, BooleanOptionalAction
from copy import deepcopy
from scipy.stats import pearsonr
from torch.nn import Module, L1Loss
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Union, Tuple, List
from functools import partial
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.utils import call, instantiate


def results_heatmap(df1:      pd.DataFrame,
                    df2:      pd.DataFrame, 
                    suptitle: str,
                    title1:   str,
                    title2:   str,
                    x_label:  str,
                    y_label:  str,
                    vertical: bool = False) -> None:
    """Display the heatmap of the two given dataframes. They should have the
    same index and columns values for a appropriate comparaison.

    Args:
        df1 (pd.DataFrame): 
            first dataframe to display

        df2 (pd.DataFrame): 
            second dataframe to display

        suptitle (str): 
            title of the whole plot

        title1 (str): 
            title for the first dataframe heatmap

        title2 (str): 
            title for the second dataframe heatmap

        x_label (str): 
            label for the x-axis (ie label of the index of the dataframes)

        y_label (str): 
            label for the y-axis (ie label of the columns of the dataframes)
        
        vertical (bool, optional): 
            if true, the heatmap will be display one above the other, 
            if False, the two heatmap will be next to each other. 
            Defaults to False.
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

def format_batch(batch: List[Tuple], phenotypes: List[str]) -> Tuple[torch.Tensor]:
    """ This function converts the output of the `getItem` method of the `SNPmarkersDataset` 
    to tuple of torch.tensor whatever the number of phenotypes set via the proprety `set_phenotypes`.
    This function is designed to be given to a Dataloader of pytorch via the collate_fn function.

    Args:
        batch (List[Tuple]): The list contains all samples selected for the current batch. 
          Each element of batch is a tuple:
          
          - The first element is the array with the input features.
          - The second element is the phenotypic value if the proprety `set_phenotypes` 
            is set with a single phenotype, otherwise a dictionary the phenotype as key and the phenotypic value as value.
        
        phenotypes (List[str]): 
            Used to define the order of the phenotypes in the target tensor if several phenotypes are used.
            In this case, should be the list used to set the proprety `set_phenotypes`.
            This argument is unused if the proprety `set_phenotypes` is set with a single phenotype. 
          
    Returns:
        Tuple[torch.Tensor]: 
            The formatted batch with the first element containing the tensor of the input feature 
            and the second element the tensor of the target values.
    """
    x = []
    y = []
    
    # If the type of the target isn't a dict, just seperate the data of the target and return
    if type(batch[0][1]) != dict:
        for item in batch:
            x.append(item[0])
            y.append(item[1])
        # Convert x and y to np.array to speed up computations
        return torch.Tensor(np.array(x)), torch.Tensor(np.array(y))
    
    else: 
        for item in batch:
            x.append(torch.Tensor(item[0]))
            y.append(torch.Tensor([item[1][key] for key in phenotypes]))
        
        return torch.stack(x), torch.stack(y)

def train_DL_model(
        model: Module,
        optimizer : Union[Optimizer, partial],
        train_dataloader: DataLoader,
        validation_dataloader: DataLoader,
        n_epoch: int,
        criterion: Module = L1Loss(),
        scheduler: Union[None, LRScheduler] = None,
        phenotype: Union[str, None] = None,
        log_wandb: bool = True,
        initial_phenotype = None,
        validation_mean: float = 0,
        validation_std: float = 1,
        early_stop_threshold: float = 0,
        early_stop_n_epoch: int = 10,
        display_evolution_threshold: float = 2.0,
        model_save_path: Union[str, None] = None, 
    ) -> None:
    """Define a basic universal training function that support wandb logging. 
    Evaluation on the validation dataset is performed every epoch.
    Models should be transferred to gpu at initialisation in order to allow network partionning on several GPU.
    Models input are initially send to cuda:0 and model output is expected on cuda:0. 
    The function as a automatic early stop if the train_loss is below 0.01 for 10 epochs 
    (ie the model as completely fit the train set and no changes are expected)

    Args:
        model (Module): The model to train.

        optimizer (Optimizer or functool.partial): 
            Optimizer used to train the given model.
            A partial optimizer (where the model parameters are the only missing parameter) 
            can be used instead of the already instansitated object.
        
        train_dataloader (DataLoader): 
            Dataloader containing the training dataset.

        validation_dataloader (DataLoader): 
            Dataloader containing the training dataset.

        n_epoch (int): 
            number of epoch to train the model

        criterion (Module, optional): 
            Pytorch class defining the loss function to be used. Defaults to L1Loss().

        scheduler (None | LRScheduler, optional): 
            Pytorch LRScheduler object to perform learning rate Scheduling. Defaults to None.
        
        phenotype (str | None, optional): 
            String containing the current phenotype studied. 
            This argument is only used for logging purposes. Defaults to None.

        log_wandb (bool, optional): 
            If true, the loss and correlation are logged into wandb, 
            otherwise they are printed after every epoch. Defaults to True.

        initial_phenotype (np.array | None, optionnal): 
            If the model compute a residual phenotype, 
            you can provide the basis to see the evolution of correlation of the final prediction. 
            Defaults to None.

        validation_mean (float, optional): 
            mean of the phenotypes from the validation set that was substratcted when normalizing the validation phenotype.
            This value will be added back to the validation target and the model prediction on the validation set 
            to be able to compare validation loss when non centered models. 
            Defaults to 0 (no recentering).
        
        validation_std (float, optional): 
            standard deviation of the phenotypes from the validation set 
            that was used as denominator when normalizing the validation phenotype. 
            This value will be added back to the validation target and the model prediction 
            on the validation set to be able to compare validation loss when non centered models. 
            Defaults to 1 (no standarization).

        early_stop_threshold (float, optional):
             percentage of the maximum correlation below the early stop counter stop incrementing. 
             This value should stay between 0 and 1 included. Defaults to 0.
            
        early_stop_n_epoch (int, optional): 
            number of consecutive epoch where the correlation is below early_stop_threshold*max_correlation. 
            Defaults to 10.
        
        display_evolution_threshold (float, optional): 
            If the modification in correlation previous_correlation - correlation is greater than 
            this value times the max_correlation, display the graph of the two computation 
            in order to see this evolution. Defaults to 2.0 (ie no logging).

        model_save_path (str, optional): 
            Path to the file where to store the best model on the validation set 
            (the extension .pth will automatically be added). 
            If None, the model isn't saved. Defaults to None.
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

    if type(scheduler) == partial:
        try:
            scheduler = scheduler(optimizer)
        except Exception as e:
            raise Exception(f"The following error occured when completing the scheduler: {e.args}")

        assert type(scheduler) != LRScheduler, \
            "The partial scheduler given don't yield a LRScheduler class when the optimizer are given"


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    
    print(f"Devices detected: {[f'cuda:{i}' for i in range(torch.cuda.device_count())] if torch.cuda.device_count() > 1 else device}")
    print(f"Model architecture : \n {model}")
    print(f"Numbers of parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Optimizer used: {optimizer}")
    if scheduler != None:
        print(f"Scheduler class used: {scheduler.__class__.__name__}")
    
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Train feature batch shape: {train_features.size()}")
    print(f"Train labels batch shape: {train_labels.size()}")
    
    val_features, val_labels = next(iter(train_dataloader))
    print(f"Validation feature batch shape: {val_features.size()}")
    print(f"Validation labels batch shape: {val_labels.size()}")

    max_correlation = 0
    early_stop_counter = 0
    train_early_stop_counter = 0
    TRAIN_EARLY_STOP_N_EPOCH = 10
    TRAIN_EARLY_STOP_THRESHOLD = 0.01

    # Initially set to 2 (an unobtainable correlation) to detect easily the first iteration
    previous_correlation = 2
    previous_predictions = []

    # Define metrics to improve the results display
    if log_wandb:
        epoch_key = "epoch"
        if type(phenotype) == str:
            # Define the keys for the dictonary used by wandb to avoid mispelling mistakes
            correlation_key = f"correlation {f'{phenotype}' if phenotype is not None else ''}"
            train_loss_key = f"train_loss {f'{phenotype}'  if phenotype is not None else ''}"
            validation_loss_key = f"validation_loss {f'{phenotype}' if phenotype is not None else ''}"
            
            wandb.define_metric(correlation_key, step_metric=epoch_key, summary='max')

        else:
            # Define the keys for the dictonary used by wandb to avoid mispelling mistakes
            correlation_key = [f"correlation {pheno}" for pheno in phenotype]
            train_loss_key = f"train_loss all phenotypes"
            epoch_key = "epoch"
            validation_loss_key = f"validation_loss all phenotypes"
            for key in correlation_key:
                wandb.define_metric(key, step_metric=epoch_key, summary='max')
        
        wandb.define_metric(train_loss_key, step_metric=epoch_key, summary='none')
        wandb.define_metric(validation_loss_key, step_metric=epoch_key, summary='none')
        wandb.define_metric(epoch_key, hidden=True, summary='none')

        wandb.config["early_stop_threshold"] = early_stop_threshold
        wandb.config["early_stop_n_epoch"] = early_stop_n_epoch
        wandb.config["display_evolution_threshold"] = display_evolution_threshold
        wandb.config["n_epoch"] = n_epoch
        wandb.config["optimizer"] = type(optimizer)
        wandb.config["criterion"] = type(criterion)
        
    for epoch in range(n_epoch):
        train_loss = []
        model.train()
        for x,y in train_dataloader:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            if len(y.shape) == 1:
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

        train_loss = np.array(train_loss).mean()
        print(f"Finished training for epoch {epoch}{f' for {phenotype}' if type(phenotype) == str and phenotype is not None else ''}.",
              f"{f'Train loss: {train_loss} ' if not log_wandb else ''}")

        val_loss = []
        predicted = []
        target = []
        with torch.no_grad():
            model.eval()
            for x,y in validation_dataloader:
                x,y = x.to(device), y.to(device)
                output = model(x)
                if len(y.shape) == 1:
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
            predicted = predicted.reshape((predicted.shape[0],) if type(phenotype) == str else (predicted.shape[0],len(phenotype))) 
            target = target.reshape((target.shape[0],) if type(phenotype) == str else (target.shape[0],len(phenotype)))

            if initial_phenotype is not None:
                initial_phenotype = initial_phenotype.reshape((initial_phenotype.shape[0], ) if type(phenotype) == str else (initial_phenotype.shape[0],len(phenotype)))
                predicted = initial_phenotype + predicted
                target = initial_phenotype + target
            
            if scheduler is not None:
                scheduler.step()
            
            with warnings.catch_warnings(record=True) as w:
                if type(phenotype) == str:
                    correlation = pearsonr(predicted, target).statistic
                else:
                    if log_wandb:
                        wandb.log({validation_loss_key: np.array(val_loss).mean()})
                    correlation = []
                    for i in range(len(phenotype)):
                        correlation.append(pearsonr(predicted[:,i], target[:,i]).statistic)
                        if log_wandb:
                            wandb.log({correlation_key[i]: correlation[-1]})

                if len(w) > 0:
                    print(f"Stop execution as model converged to outputing always the same value ({predicted[0]})")
                    for warning in w:
                        print(warning)
                    if log_wandb:
                        wandb.finish(1)
                    exit(1)
                
            if log_wandb and type(phenotype) == str:
                wandb.log({validation_loss_key: np.array(val_loss).mean(), correlation_key: correlation,})
            
            print(f"Validation step for epoch {epoch}{f' for {phenotype}' if type(phenotype) == str and phenotype is not None else ''} finished!",
                f"{f' Correlation: {correlation}. Validation loss: {np.array(val_loss).mean()}' if not log_wandb else ''}")
            
                                
            # Use the sum of the correlation as the stopping criterion for multitrait regression
            if type(phenotype) != str:
                correlation = sum(correlation)

            if correlation > max_correlation:
                max_correlation = correlation
                if model_save_path != None:
                    # Use deepcopy as state_dict() only returns a shallow copy
                    best_model_state = deepcopy(model.state_dict())

            if log_wandb:

                if previous_correlation - correlation > display_evolution_threshold * max_correlation and \
                        previous_correlation <= 1 and previous_correlation >= -1:
                    wandb.log({"visualization": wandb.Image(compare_correlation(
                            previous_predictions,
                            predicted,
                            target,
                            epoch
                        ))})
                    
                previous_correlation = correlation
                previous_predictions = predicted


            # ---------------------------------- Early stop management --------------------------------------------
            
            # Take the absolute value such that small oscillation arround zero doesn't reset the counter
            if abs(correlation) < early_stop_threshold * max_correlation:
                if early_stop_counter == 0:
                    print(f"Start early stop counter, current{' summed' if type(phenotype) != str else ''} correlation threshold is {early_stop_threshold * max_correlation}")
                early_stop_counter +=1
            else:
                early_stop_counter = 0
            
            if early_stop_counter >= early_stop_n_epoch:
                print(f"Early stop condition on correlation met. Best{' summed' if type(phenotype) != str else ''} correlation observed: {max_correlation}")
                break

            if train_loss < TRAIN_EARLY_STOP_THRESHOLD:
                train_early_stop_counter += 1
            
            if train_early_stop_counter >= TRAIN_EARLY_STOP_N_EPOCH:
                print(f"Early stop condition on train loss met. Best{' summed' if type(phenotype) != str else ''} correlation observed: {max_correlation}")
                break
    
    if model_save_path != None:
        torch.save(best_model_state, model_save_path + ".pth")


def print_elapsed_time(start_time: float) -> str:
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

def get_partial_optimizer(optimizer_name: str, **kwargs) -> functools.partial:
    """Dummy function use to produced a partial optimizer from a hydra config file. 
    This solution enable to consider 
    the optimization class not as a string (due to hydra) but as the function to be called.

    Args:
        optimizer_name (str): Name of the optimizer to use (should be a optimizer class of torch.optim)

    Returns:
        functools.partial: a partial function waiting the model parameters to yield the torch optimizer.
    """
    return functools.partial(getattr(torch.optim, optimizer_name), **kwargs)

def get_partial_scheduler(scheduler_name: str, **kwargs) -> functools.partial:
    """Dummy function use to produced a partial scheduler from a hydra config file. 
    This solution enable to consider 
    the optimization class not as a string (due to hydra) but as the function to be called.

    Args:
        scheduler_name (str): Name of the optimizer to use (should be a class of torch.optim)

    Returns:
        functools.partial: a partial function waiting the optimize to yield the torch scheduler.
    """
    return functools.partial(getattr(torch.optim.lr_scheduler, scheduler_name), **kwargs)

def train_from_config(
        phenotype: Union[str, list],
        run_cfg: DictConfig,
        train_dataset: Dataset = None,
        validation_dataset: Dataset = None,
        model: Module = None,
        model_save_path: Union[str, None] = None,
        **kwargs):
    """Launch the training based on the parameters from the hydra config files. 
    Custom datasets and model can be provided with the optionals arguments 
    (In the case they depend they have an argument that depend on the data for example).

    Args:
        phenotype (str): 
            phenotype on which the model should be trained on (should be a key of SNPmarkersDataset.phenotypes).
        
        run_cfg (DictConfig): 
            DictConfig object build from the hydra config. 
            This object should be the output of the `hydra.compose` function based on the arguments of the script.

        train_dataset (Dataset, optional): 
            the ready to use train dataset object to use. 
            If None, a default one will be created from the config. Defaults to None.
        
        validation_dataset (Dataset, optional): 
            the ready to use validation dataset object to use. 
            If None, a default one will be created from the config. Defaults to None.

        model (Module, optional): 
            the model object to train. 
            If None, a default one will be created from the config. Defaults to None.

        model_save_path (str, optional): 
            path where to save the resulting model 
            (the extension .pth will automatically be added). Default to None (no saving).
    """

    if model == None:
        if type(phenotype) == list:
            model = instantiate(run_cfg.model_config.model, output_size = len(phenotype))
        else:
            model = instantiate(run_cfg.model_config.model)

    if train_dataset == None:
        train_dataset= instantiate(run_cfg.data.train_dataset)
        train_dataset.set_phenotypes = phenotype
    if validation_dataset == None:
        validation_dataset = instantiate(run_cfg.data.validation_dataset)
        validation_dataset.set_phenotypes = phenotype
    
    if type(phenotype) == list:
        call(run_cfg.train_function_config,
            phenotype = phenotype, 
            model= model,
            train_dataloader = instantiate(run_cfg.train_function_config.train_dataloader, dataset=train_dataset, collate_fn=partial(format_batch, phenotypes=phenotype)),
            validation_dataloader = instantiate(run_cfg.train_function_config.validation_dataloader, dataset=validation_dataset, collate_fn=partial(format_batch, phenotypes=phenotype)),
            model_save_path = model_save_path,
            **kwargs)
    else:
        call(run_cfg.train_function_config,
            phenotype = phenotype, 
            model= model,
            train_dataloader = instantiate(run_cfg.train_function_config.train_dataloader, dataset=train_dataset),
            validation_dataloader = instantiate(run_cfg.train_function_config.validation_dataloader, dataset=validation_dataset),
            model_save_path = model_save_path,
            **kwargs)

def list_of_strings(arg):
    """Function defining a custom class for argument parsing."""
    return arg.split(',')

def get_clean_config(run_cfg: DictConfig) -> dict:
    """Create a clean dictionary can only contain useful configuration infomations on the current run based on the given hydra config.

    Args:
        run_cfg (DictConfig): hydra config file fetch using the compose API.

    Returns:
        dict: clean dictionary usable as config for the wandb run.
    """
    
    # Define custom resolver for the computation of the hidden_nodes for Deep_MLP (See https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html)
    OmegaConf.register_new_resolver("mul", lambda x, y: int(x * y), replace= True)
    OmegaConf.register_new_resolver("div", lambda x, y: int(x / y), replace= True)

    wandb_cfg = OmegaConf.to_container(run_cfg, resolve=True)
    for k,v in wandb_cfg["model_config"].items():
        wandb_cfg[k] = v

    # Remove the layer "model_config as it's only caused by the structure of config files and isn't useful"
    wandb_cfg.pop("model_config")
    wandb_cfg.pop("train_function_config")
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
                         vertical:    bool = False) -> None:
    """Display the evolution of two given metrics given a single parameter.

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

def compare_correlation( prediction_before:     list,
                         prediction_after:      list, 
                         target:                list,
                         epoch:                 int) -> None:
    """Display the comparaison of two sucessive output of the model.

    Args:
        prediction_before (list): first output of the model.
        prediction_after (list): output of the next epoch of the model.
        target (list): True values of the validation set.
        epoch (int): Epoch of the prediction_after.
    """
    
    subtitle_font =  {"size": 14}
    title_font = {"weight": "bold" ,"size": 18}
    
    fig,(ax1, ax2) = plt.subplots(1,2, figsize=(17 , 6.5))

    fig.suptitle(f"Evolution of the predicted values (on the validation set) between epoch {epoch -1} and {epoch}", font=title_font)
    sns.scatterplot(x=target, y=prediction_before, ax=ax1)
    plt.yticks(rotation=0)
    ax1.plot(np.unique(target), 
         np.poly1d(np.polyfit(target, prediction_before, 1))
         (np.unique(target)), color='red')
    ax1.set_title(f"Epoch {epoch - 1}. Correlation: {pearsonr(target, prediction_before).statistic:.5f}", font=subtitle_font)
    ax1.set_xlabel("Target")
    ax1.set_ylabel("Predictions")
    ax1.tick_params(rotation=0)

    sns.scatterplot(x=target, y=prediction_after, ax=ax2)
    ax2.plot(np.unique(target), 
         np.poly1d(np.polyfit(target, prediction_after, 1))
         (np.unique(target)), color='red')
    ax2.set_title(f"Epoch {epoch}. Correlation: {pearsonr(target, prediction_after).statistic:.5f}", font=subtitle_font)
    ax2.set_xlabel("Target")
    ax2.set_ylabel("Predictions")
    ax2.tick_params(rotation=0)
    
    # Draw the figure as it's not display
    fig.canvas.draw()

    # Convert the image into an np array for the display via wandb
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data

def get_default_config_parser() -> ArgumentParser:
    """ Create a default argument parser with the appropriate format for the scripts

    Returns:
        ArgumentParser: The ready to use default argument parset.
    """
    parser = ArgumentParser()

    parser.add_argument("--model", "-m", required=True, type=str, help="Name of the file (without file extention) to define the model to train (should be found in configs/model_config)")
    parser.add_argument("--data", "-d", required=True, type=str, help="Name of the file (without file extention) to use for the data (should be found in configs/data)")
    parser.add_argument("--phenotypes", "-p", required=True, type=list_of_strings, help="Phenotype(s) to perform the sweep (format example: ep_res,de_res,size_res)")
    parser.add_argument("--wandb_run_name", "-w", required=False, type=str, help="String to use for the wandb run name")
    parser.add_argument("--train_function", "-f", required=True, type=str, help="Name of the file (without file extention) to use to create the training function (should be found in configs/train_function_config)")
    parser.add_argument("--all", default=False, action=BooleanOptionalAction, help="If True, perform multi-trait regression on all given phenotypes")
    parser.add_argument("-o", "--output_path", default=None, type=str, help="Path to store the resulting model")

    return parser

def convert_categorical_to_frequency(data: List[List[int]], path = "../Data/gptranformer_embedding_data.json") -> np.ndarray:
    """Function that convert the catergorical encoding to the frequency one based on the precomputed statistics.

    Args:
        data (List[List[int]]):
            The data to convert. The first dimention defines the number of individuals. For every individual, the SNP markers should be provided. 
        path (str, optional): 
            Path to the precomputed statistics. 
            Defaults to "../Data/gptranformer_embedding_data.json".

    Returns:
        np.ndarray: Converted dataset for categorical to frequency.
    """
    with open(path,"r") as f:
        freq_data = json.load(f)
    
    results = []
    for sample in data:
        func = lambda t: [freq_data[str(t[0])]["p"]**2, 2*freq_data[str(t[0])]["p"]*freq_data[str(t[0])]["q"],freq_data[str(t[0])]["q"]**2].__getitem__(t[1])
        results.append(list(map(func, enumerate(sample))))
    return np.array(results, dtype=np.float32)
