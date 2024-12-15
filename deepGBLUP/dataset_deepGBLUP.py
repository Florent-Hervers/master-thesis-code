import numpy as np 
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from utils import  make_G_D_E, mixed_model, call_Z


import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import SNPmarkersDataset


class SNPDataset(Dataset):
    def __init__(self, X, a, d, e, ids, y=None):
        self.X = X
        self.y = y
        self.ids = ids
        self.a = a 
        self.d = d
        self.e = e

    def __len__(self):
        return len(self.ids)
    

    def __getitem__(self,idx):
        if self.y is not None:
            return {
                'X':self.X[idx], 'y':self.y[idx], 
                'a':self.a[idx], 'd':self.d[idx], 'e':self.e[idx],
                'id':self.ids[idx]
            }
        else:
            return {
                'X':self.X[idx],
                'a':self.a[idx], 'd':self.d[idx], 'e':self.e[idx],
                'id':self.ids[idx]
            }

def load_dataset(raw_path, phen_path, h2, device='cpu', phenotype = "ep_res"):

    train_dataset = SNPmarkersDataset("train")
    validation_dataset = SNPmarkersDataset("validation")

    train_dataset.set_phenotypes = phenotype
    validation_dataset.set_phenotypes = phenotype

    # to tensor
    train_X = torch.tensor(train_dataset.get_all_SNP().values, dtype= torch.float32)
    train_y = torch.tensor(train_dataset.phenotypes[phenotype].values, dtype=torch.float32)
    #train_ids = np.array(train_ids,dtype=str)
    test_X = torch.tensor(validation_dataset.get_all_SNP().values, dtype=torch.float32)
    #test_ids = np.array(test_ids,dtype=str)
    test_y = torch.tensor(validation_dataset.phenotypes[phenotype].values, dtype=torch.float32)

    # Print some infos to check the validity of spliting
    print(f"Example of train sample: {train_X[0]}")
    print(f"Example of validation sample: {test_X[0]}")
    print(f"Train_X shape: {train_X.shape}, test_X shape: {test_X.shape}")
    print(f"Train_y shape: {train_y.shape}, test_y shape: {test_y.shape}")
    
    # cal genetic effects
    X = torch.cat([train_X, test_X], dim=0)
    y = train_y.clone()
    Gi, Di, Ei = make_G_D_E(X, invers=True, device = device)
    Z = call_Z(len(train_X), len(train_X)+len(test_X))
    glamb = (1 - h2)/h2
    dlamb = elamb =  (1 - h2*0.1)/(h2*0.1)
    
    a = mixed_model(Z, Gi, glamb,y)
    d = mixed_model(Z, Di, dlamb,y)
    e = mixed_model(Z, Ei, elamb,y)

    train_a, test_a = a[:len(train_X)], a[len(train_X):]
    train_d, test_d = d[:len(train_X)], e[len(train_X):]
    train_e, test_e = e[:len(train_X)], d[len(train_X):]
    
    train_dataset = SNPDataset(train_X,  train_a, train_d, train_e, np.zeros(len(train_X)), train_y)
    test_dataset = SNPDataset(test_X, test_a, test_d, test_e, np.zeros(len(test_X)), test_y)
    return train_dataset, test_dataset, X.shape[1], y.mean()



