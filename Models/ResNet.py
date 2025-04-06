import math
import torch
import torch.nn.functional as F

from torch import nn
from typing import List
from omegaconf import ListConfig
from Models.LCLNN import LocalLinear
from Models.SNPEncoder import SNPEncoder
from Models.VariableSizeOutputModel import VariableSizeOutputModel

class ResNetBlock(nn.Module):
    def __init__(self, kernel_size:int , input_channel_size: int, nlayers: int, reduce_first_layer: bool = False):
        super(ResNetBlock, self).__init__()
        
        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        if reduce_first_layer:
            output_channel_size = input_channel_size * 2
            first_layer_stride = 2
            self.residual_conv = nn.Conv1d(input_channel_size, output_channel_size, 1, stride=2)
        else:
            output_channel_size = input_channel_size
            first_layer_stride = 1
            self.residual_conv = None
        
        layers = [
            nn.Conv1d(input_channel_size, output_channel_size, kernel_size, padding=1, stride=first_layer_stride),
            nn.BatchNorm1d(output_channel_size)
        ]
        
        for _ in range(1, nlayers + 1):
            layers.append(nn.ReLU())
            layers.append(nn.Conv1d(output_channel_size, output_channel_size, kernel_size, padding=1))
            layers.append(nn.BatchNorm1d(output_channel_size))

        self.block = nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        y = self.block(x)

        if self.residual_conv:
            x = self.residual_conv(x)
        
        return F.relu(y + x)
    

class ResNet(VariableSizeOutputModel):
    valid_aggregation = ["pooling", "strided", "strided_LCL"]

    def __init__(self, start_channel_size: int, kernel_size: int, layers_size: List[int], regressor_hidden_size: List[int], aggregation: str = "pooling", encoding: str = None, **kwargs):
        super(ResNet,self).__init__(**kwargs)

        # In the case where the parameters are given via hydra configs file, the type of arguments are ListConfig and not list.
        if type(regressor_hidden_size) == ListConfig:
            regressor_hidden_size = list(regressor_hidden_size)

        if type(layers_size) == ListConfig:
            layers_size = list(layers_size)

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        if aggregation not in self.valid_aggregation:
            raise Exception(f"Valid aggregation values are {self.valid_aggregation} but got {aggregation} instead!")
        
        self.encoder = None
        if encoding != None:
            self.encoder = SNPEncoder(mode = encoding).to(self.device)

        self.conv = nn.Conv1d(1, start_channel_size, 7, stride=2, padding=3).to(self.device)
        
        if aggregation == self.valid_aggregation[0]:
            self.aggreg1 = nn.MaxPool1d(3, stride=2, padding=1).to(self.device)
        elif aggregation == self.valid_aggregation[1] or aggregation == self.valid_aggregation[2]:
            self.aggreg1 = nn.Conv1d(start_channel_size, start_channel_size, 3, stride=2, padding=1).to(self.device)

        layers = [
            ResNetBlock(kernel_size, start_channel_size, layers_size[0])
        ]

        for i, size in enumerate(layers_size[1:]):
            layers.append(
                ResNetBlock(kernel_size, start_channel_size * (2**i), size, reduce_first_layer=True)
            )
        
        self.layers = nn.Sequential(*layers).to(self.device)

        if aggregation == self.valid_aggregation[0]:
            self.aggreg2 = nn.AdaptiveAvgPool1d(1).to(self.device)
            regressor_input_size = start_channel_size * (2**(len(layers_size) - 1))
        elif aggregation == self.valid_aggregation[1]:
            self.aggreg2 = nn.Conv1d(start_channel_size * (2**(len(layers_size)-1)), 1, 1).to(self.device)
            regressor_input_size = math.ceil(36304 / (2**(len(layers_size) + 1)))
        elif aggregation == self.valid_aggregation[2]:
            regressor_input_size = math.ceil(36304 / (2**(len(layers_size) + 1)))
            self.aggreg2 = LocalLinear(
                in_channels = start_channel_size * (2**(len(layers_size)-1)),
                out_channels = 1,
                kernel_size = 1,
                in_features = regressor_input_size,
            ).to(self.device)

        self.flatten = nn.Flatten().to(self.device)
        
        if len(regressor_hidden_size) == 0:
            raise Exception("An empty list isn't a valid input for the regressor_hidden_size parameter.")
        
        mlp_layers = [nn.Linear(regressor_input_size, regressor_hidden_size[0])]
        for i in range(1, len(regressor_hidden_size)):
            mlp_layers.append(nn.ReLU())
            mlp_layers.append(nn.Linear(regressor_hidden_size[i-1], regressor_hidden_size[i]))
        mlp_layers.append(nn.ReLU())
        mlp_layers.append(nn.Linear(regressor_hidden_size[-1], self.output_size))

        self.output = nn.Sequential(*mlp_layers).to(self.device)

    def forward(self, x):

        if self.encoder != None:
            x = self.encoder(x)

        if len(x.shape) == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        
        if len(x.shape) != 3:
            raise Exception(f"Bad dimention for the input, only 2 or 3 dimentional input are allowed but got {len(x.shape)} instead!")

        x = self.conv(x)
        x = self.aggreg1(x)
        x = self.layers(x)
        x = self.aggreg2(x)
        x = self.flatten(x)
        return self.output(x)