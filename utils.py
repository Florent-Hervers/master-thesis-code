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
