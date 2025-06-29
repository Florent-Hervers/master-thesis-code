import os
import time

import torch
import wandb
from torch.utils.data import DataLoader

from train_test import train_model, test_model
from model import deepGBLUP
from dataset_deepGBLUP import load_dataset

################ CONFIG ####################
# data path
raw_path = '../Data/BBBDL_BBB2023_MD.raw' # path of raw file
phen_path = '../Data/BBBDL_pheno_2023bbb_0twins_6traits_mask' # path of phenotype file
bim_path = '../Data/BBBDL_BBB2023_MD.bim' # optional: path of bim file to save SNP effects. If you don't have bim file just type None 

# train cofig
lr =  5e-4 # list of cadidate learning rate
epoch = 350   # max value of cadiate epoch
batch_size = 64 # train batch size

device = 'cuda' # type 'cpu' if you use cpu device, or type 'cuda' if you use gpu device.
h2 = [0.30533, 0.340504, 0.379522, 0.330629, 0.432115, 0.382769]

# save config
save_path = 'test_GBLUP' # path to save results

##############################################

####################################################
##                      Caution                   ## 
##  Users unfamiliar with python and pytorch      ##
##  should not modify the code below.             ##
####################################################
# os.makedirs(save_path, exist_ok=True)

wandb.init(
    project = "TFE",
    config = {
        "model_name": "deepGBLUP",
        "batch size": batch_size,
        "learning_rate": lr,
        "nb epochs": epoch,
        "h2": h2,
    },
    name = "Test corrected deepGBLUP with correct heritability",
    tags = ["debug"],
)


s_time = time.time()

for i, phenotype in enumerate(["ep_res", "de_res", "FESSEp_res", "FESSEa_res", "size_res", "MUSC_res"]):
    # Load Dataset
    train_dataset, test_dataset, num_snp, ymean = load_dataset(raw_path, phen_path, h2[i], device, phenotype)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = deepGBLUP(ymean, num_snp).to(device)

    # Train model
    train_time  = train_model(
        model, train_dataloader, test_dataloader, 
        save_path, device, lr, epoch, h2[i], save_model = False, phenotype=phenotype
    )

"""
# test model
test_time = test_model(
        model, 
        test_dataloader,
        device, save_path
        )

# Save hyperparameters
with open(os.path.join(save_path,'setting.txt'),'w') as save:
    save.write(f'Path of raw file: {raw_path}\n')
    save.write(f'Path of phenotype file: {phen_path}\n')
    save.write('-'*50+'\n')
    save.write(f'learning rate: {lr}\n')
    save.write(f'epoch: {epoch}\n')
    save.write(f'batch_size: {batch_size}\n')
    save.write(f'Device: {device}\n')
    save.write('-'*50+'\n')
    save.write(f'H2: {h2}\n')
    save.write('train time\t'+str(train_time)+'\n')
    save.write('test time\t'+str(test_time)+'\n')
"""  
