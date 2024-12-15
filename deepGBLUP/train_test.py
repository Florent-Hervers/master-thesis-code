import os, time

import numpy as np
import torch.optim as optim
import torch
import wandb
from scipy.stats import pearsonr

from tqdm import tqdm
from utils import perdictive_ability

def train_model(model, train_dataloader, test_dataloader, save_path, device, lr, epoch, h2, save_model, phenotype):

   
    model._init_weights()
    
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    """
    log = open(os.path.join(save_path,'log.txt'),'w')
    log.close()
    """
    s_time = time.time()
    model.train()
    train_loss = []
    for e in range(epoch):
        for i, data in enumerate(train_dataloader):

            # forward
            a, d, e_  =  data['a'].to(device), data['d'].to(device), data['e'].to(device)
            X = data['X'].to(device)
            true_y = data['y'].to(device)
            pred_y = model(X) 

            pa = perdictive_ability(pred_y, true_y, h2)
            pred_y = pred_y +a +d +e_
            loss = (pred_y - true_y).abs().mean()

            train_loss.append(loss.cpu().detach())
            """
            monitor = f'epoch {e+1} ({i+1}/{len(train_dataloader)}): loss ({round(float(loss),3)}), p.a ({round(float(pa),3)})'
            print(monitor+' '*10,end='\r')
            """

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #print(monitor+' '*10)

        """
        log = open(os.path.join(save_path,'log.txt'),'a')
        log.write(f'{monitor}\n')
        log.close()
        """
        wandb.log({
                "epoch": e, 
                f"train_loss {phenotype}": np.array(train_loss).mean()
            }
        )
        test_model(model, test_dataloader, device, save_path, e, phenotype)

    running_time = time.time() - s_time

    # save last parameters and load best paramters
    if save_model:
        torch.save(model.state_dict(), os.path.join(save_path,f'last.pth'))
    del loss, optimizer
    return running_time



def test_model(model, test_dataloader, device, save_path, epoch, phenotype):


    model.eval()
    
    s_time = time.time()
    pred_y = []
    ids = []
    print(f'Testing for epoch {epoch}...')
    val_loss = []
    true_y = []

    for data in tqdm(test_dataloader):
        X = data['X'].to(device)
        a, d, e  =  data['a'].to(device), data['d'].to(device), data['e'].to(device)
        with torch.no_grad():
            batch_pred_y = model(X)
        batch_pred_y = batch_pred_y +a +d +e
        pred_y.append(batch_pred_y)
        true_y.append(data["y"])
        loss = (batch_pred_y.cpu().detach() - data["y"]).abs().mean()
        val_loss.append(loss)
        
        ids.append(data['id'])

    pred_y = torch.cat(pred_y).cpu().detach()
    true_y = torch.cat(true_y)
    ids = np.concatenate(ids)
    wandb.log({
            "epoch": epoch, 
            f"validation_loss {phenotype}": np.array(val_loss).mean(),
            f"correlation {phenotype}": pearsonr(pred_y, true_y).statistic,
        }
    )
    running_time = time.time() - s_time

    """
    with open(os.path.join(save_path,f'sol.txt'), 'w') as save:
        save.write('IDS\tTRUE\tdeepGBLUP\n')
        for id,  py in zip(ids, pred_y):
            save.write(f'{id}\t{py}\n')
    """
    return running_time

