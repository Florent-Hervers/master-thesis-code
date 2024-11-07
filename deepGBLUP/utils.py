import torch
import math
import numpy as np
import os

from tqdm import tqdm


def mixed_model(Z, A, lamb, y):
        """ Compute the GBLUP value as described in formula (2) of the paper

        Args:
            Z (torch.Tensor): incidance matrix (should be computed using the call_Z function)
            A (torch.Tensor): inverse of the genomic relationship matrix to use (should be one of the output of the make_G_D_E function with invert = True)
            lamb (float): normalisation scalar (Lambda in the formula (2) of the paper)
            y (torch.Tensor): phenotype array for the training samples.

        Returns:
            torch.Tensor: The genomic value predicted by GBLUP and the given genomic relationship matrix
        """
        y = y - y.mean()
        e  = torch.inverse(Z.T @ Z + A *lamb) @ Z.T @ y
        return e

def call_Z(ref_len, whole_len):
    """ Create the incidance matrix Z described in the paper (ie a matrix with 1 on the diagonale and zeros otherwise)

    Args:
        ref_len (int): size of the training_set
        whole_len (int): size of the whole dataset (training + test)

    Returns:
        torch.Tensor: incidance matrix
    """
    Z = torch.zeros((ref_len, whole_len),dtype=torch.float32)
    for i in range(ref_len): Z[i,i] = 1
    return Z

def make_G_D_E(X, invers=True, device='cpu'):
    """Compute the genomic relationship matrixes

    Args:
        X (torch.tensor): genotype data (should be computed in the additive way: {0,1,2})
        invers (bool, optional): If True, the function will output the inverse of the genomic relationship matrixes. Defaults to True.
        device (str, optional): Not used. Defaults to 'cpu'.

    Returns:
        torch.Tensor, torch.Tensor, torch.Tensor: the genomic relationship matrixes G, D and E
    """

    # cal freq matrix
    n,k = X.shape
    pi = X.sum(0)/(2*n) # X.mean(0)/2
    P = (pi).unsqueeze(0) 

    # make A (dummay pedigree)
    A = torch.eye(len(X))

    # make G
    Z = X - 2*P 
    G = (Z @Z.T)  /(2*(pi*(1-pi)).sum()) 

    # make D
    print('make dominance matrix')
    W = X.clone()
    for j in tqdm(range(W.shape[1])):
        W_j = W[:,j]
        W_j[W_j == 0] = -2*(pi[j]**2)
        W_j[W_j == 1] = 2*(pi[j]*(1-pi[j]))
        W_j[W_j == 2] = -2*((1-pi[j])**2)

    D = (W @W.T)  /(((2*pi*(1-pi))**2).sum()) 

    # make E
    print('make interaction marker')
    M = X - 1 
    E = 0.5*((M @ M.T) * (M @ M.T))  - 0.5*((M * M) @ (M * M).T) 
    E = E/(torch.trace(E)/n) 
    # E = D
    del W, M

    # rescaling with dummy A
    G = G * 0.99 + A * 0.01
    D = D * 0.99 + A * 0.01
    E = E * 0.99 + A * 0.01

    if invers:
        print("Inversing matrixes")
        return torch.inverse(G), torch.inverse(D), torch.inverse(E)
    return G, D, E



def GBLUP(train_X, test_X, train_y, test_y, h2):

    X = torch.cat([train_X, test_X], dim=0)
    train_len = len(train_y)

    # Compute G
    M = X - 1
    pi = X.mean(0)/2
    P = 2*(pi-0.5)
    M = M - P 
    G = (M @M.T)/(2*(pi*(1-pi)).sum())

    G_inv = torch.inverse(G)
    Z = torch.zeros((train_len, len(G_inv)),dtype=torch.float32)
    for i in range(train_len): Z[i][i] = 1
    y_c = (train_y - train_y.mean()).unsqueeze(1)
    y_c = train_y.unsqueeze(1)
    lamb = h2/(1 - h2)
    
    y_gblup = torch.inverse(Z.T @ Z + G_inv*lamb) @ Z.T @ y_c
    # m1 = perdictive_ability(test_y, y_gblup[train_len:][:,0], h2)


    return y_gblup[:,0][:train_len], y_gblup[:,0][train_len:], Z


def perdictive_ability(true, pred, h2=1):
    """Compute the predictive ability used as metric in the paper
    Args:
        true (torch.Tensor): True value of the phenotype.
        pred (torch.Tensor): predictions of the model.
        h2 (int, optional): heritability of the phenotype. Defaults to 1.

    Returns:
        float : predictive ability (definied as pearson correletion / sqrt(h2))
    """
    
    pred_mean = pred.mean()
    true_mean = true.mean()

    f1 = torch.sum((pred - pred_mean) * (true - true_mean))
    f2 = torch.sqrt(torch.sum((pred - pred_mean)**2) * torch.sum((true - true_mean)**2))
    if f2 == 0:
        return 0

    cor = f1/f2 
    return float(cor/math.sqrt(h2))

