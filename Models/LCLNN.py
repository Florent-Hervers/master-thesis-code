from omegaconf import ListConfig
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class LocalLinear(nn.Module):
    def __init__(self,in_features,local_features,kernel_size,stride=1,bias=True):
        super(LocalLinear, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size - 1

        fold_num = (in_features+self.padding -self.kernel_size)//self.stride+1
        self.weight = nn.Parameter(torch.randn(fold_num,kernel_size,local_features))
        self.bias = nn.Parameter(torch.randn(fold_num,local_features)) if bias else None

        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0.0)

    def forward(self, x:torch.Tensor):
        x = F.pad(x,[0, self.padding],value=0)
        x = x.unfold(-1,size=self.kernel_size,step=self.stride)
        x = torch.matmul(x.unsqueeze(2),self.weight).squeeze(2)+self.bias
        return x.squeeze(2)

class LCLNN(nn.Module):
    def __init__(self, 
                 num_snp, 
                 mlp_hidden_size = None
                 ):
        """ 
        Generate a standalone LCL network proposed orginally in the deepBLUP architecture.

        Args:
            num_snp (int): size of the snp sequence
            mlp_hidden_size (int, optional): Size of the hidden values in the mlp. Defaults to None.
        """
        super(LCLNN, self).__init__()

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.encoder = nn.Sequential(
            LocalLinear(num_snp, 1, kernel_size=7,stride=1),
            nn.LayerNorm(num_snp),
            nn.ReLU(),
            LocalLinear(num_snp, 1, kernel_size=5,stride=1),
            nn.LayerNorm(num_snp),
            nn.ReLU(),
            LocalLinear(num_snp, 1, kernel_size=3,stride=1),
        ).to(self.device)         

        # In the case where the parameters are given via hydra configs file, the type of the mlp_hidden_size is a ListConfig and not a list.
        if type(mlp_hidden_size) == ListConfig:
            mlp_hidden_size = list(mlp_hidden_size)


        if mlp_hidden_size == None:
            self.mlp = nn.Linear(num_snp,1).to(self.device)
        if type(mlp_hidden_size) == list:
            if len(mlp_hidden_size) == 0:
                raise Exception("An empty list isn't a valid input for the mlp_hidden_size parameter.")
            layers = [nn.Linear(num_snp, mlp_hidden_size[0])]
            for i in range(1, len(mlp_hidden_size)):
                layers.append(nn.ReLU())
                layers.append(nn.Linear(mlp_hidden_size[i-1], mlp_hidden_size[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Linear(mlp_hidden_size[-1], 1))

            self.mlp = nn.Sequential(*layers).to(self.device)
        
        else:
            self.mlp = nn.Sequential(
                nn.Linear(num_snp, mlp_hidden_size),
                nn.ReLU(),
                nn.Linear(mlp_hidden_size, 1)
            ).to(self.device)
        
        self._init_weights()
        
    def _init_weights(self):
        # weight init
        for m in self.modules():
            if isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self,X):

        X = self.encoder(X) + X
        b = self.mlp(X)
        
        return b