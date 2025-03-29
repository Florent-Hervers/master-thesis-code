import torch
import torch.nn.functional as F

from torch import nn
from omegaconf import ListConfig

class LocalLinear(nn.Module):
    def __init__(self,in_features,out_channels,kernel_size,stride=1, bias=True, in_channels = 1,):
        """Create a LCL layer. Inputs should be of size (N, C_in, L) and the output will be of size (N, C_out, L // S) where 
        - N = batch_size
        - C_in = in_channels
        - L = in_features
        - C_out = out_channels
        - S = stride

        Args:
            in_features (int): Size of the last dimention of the input.
            out_channels (int): Channel dimention of the output.
            kernel_size (int): Size of the kernel using in the layer.
            stride (int, optional): Value for the stride to use. Defaults to 1.
            bias (int, optional): If true, add bias parameters added after the computation of the kernel. Defaults to True.
            in_channels (int, optional): Number of channels of the input. Defaults to 1.
        """
        super(LocalLinear, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size - 1
        self.in_channels = in_channels
        self.in_features = in_features
        self.out_channels = out_channels

        fold_num = (in_features+self.padding -self.kernel_size) // self.stride+1
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, fold_num,kernel_size, 1))
        self.bias = nn.Parameter(torch.randn(out_channels, in_channels, fold_num, 1)) if bias else None

        nn.init.xavier_uniform_(self.weight)
        if bias :
            nn.init.constant_(self.bias, 0.0)

    def _lcl1d(self, x:torch.Tensor, weights, bias):
        x = F.pad(x,[0, self.padding], value=0)
        x = x.unfold(-1,size= self.kernel_size,step= self.stride)
        x = torch.matmul(x.unsqueeze(2), weights).squeeze(2)
        if bias != None:
            x += bias
        
        return x.squeeze(2)

    # Modified version of the implementation provided in https://d2l.ai/chapter_convolutional-neural-networks/channels.html
    def _lcl1d_multi_in(self, x, weigths, bias):
        return sum(self._lcl1d(x[:,i], weigths[i], bias[i] if bias != None else None) for i in range(x.shape[1]))
    
    def forward(self, x:torch.Tensor):
        return torch.stack([self._lcl1d_multi_in(x, self.weight[i], self.bias[i] if self.bias != None else None) for i in range(self.weight.shape[0])], 1)
    
    def extra_repr(self):
        return f"kernel_size={self.kernel_size}, stride={self.stride}, bias={True if self.bias != None else False}, in_features={self.in_features}, in_channels={self.in_channels}, out_channels={self.out_channels}"

class LCLNN(nn.Module):
    def __init__(self, 
                 num_snp, 
                 mlp_hidden_size = None,
                 dropout = 0,
                 ):
        """ 
        Generate a standalone LCL network proposed orginally in the deepBLUP architecture.

        Args:
            num_snp (int): size of the snp sequence
            mlp_hidden_size (int, optional): Size of the hidden values in the mlp. Defaults to None.
            dropout (float, optional): Probability of dropout used in the whole network. Defaults to 0.
        """
        super(LCLNN, self).__init__()

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.encoder = nn.Sequential(
            nn.Dropout(dropout),
            LocalLinear(num_snp, 1, kernel_size=5,stride=1),
            nn.LayerNorm(num_snp),
            nn.ReLU(),
            nn.Dropout(dropout),
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
            
            if mlp_hidden_size[0] <= 0:
                raise Exception(f"First hidden size must be a non negative integer value but got {mlp_hidden_size[0]} instead!")
            
            layers = [nn.Dropout(dropout), nn.Linear(num_snp, mlp_hidden_size[0])]
            
            for i in range(1, len(mlp_hidden_size)):
                # If hidden size == 0, remove the "empty" layer and continue the mlp creation to enable variating size mlp while tuning
                if mlp_hidden_size[i] == 0:
                    mlp_hidden_size[i] = mlp_hidden_size[i-1]
                    continue
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(mlp_hidden_size[i-1], mlp_hidden_size[i]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(mlp_hidden_size[-1], 1))

            self.mlp = nn.Sequential(*layers).to(self.device)
        
        else:
            self.mlp = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_snp, mlp_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
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

        if len(X.shape) == 2:
            X = X.view(X.shape[0], 1, X.shape[1])

        X = self.encoder(X) + X

        X = X.view(X.shape[0], -1)
        b = self.mlp(X)
        
        return b