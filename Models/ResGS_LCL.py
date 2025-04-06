import torch
import math

from torch import nn
from Models.LCLNN import LocalLinear
from Models.VariableSizeOutputModel import VariableSizeOutputModel

class Conv1d_BN(nn.Module):
    def __init__(self, input_size, nb_filter, kernel_size, strides=1, padding = 1):
        super(Conv1d_BN, self).__init__()
        self.conv = nn.Conv1d(input_size, nb_filter, kernel_size, padding= padding, stride=strides)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(nb_filter)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        return x

class Res_Block(nn.Module):
    def __init__(self, input_size, nb_filter, kernel_size, strides=1):
        super(Res_Block, self).__init__()
        self.block = Conv1d_BN(input_size,nb_filter=nb_filter,kernel_size=kernel_size,strides=strides)
    
    def forward(self, x):
        x = x + self.block(x)
        return x

class ResGSModel_LCL(VariableSizeOutputModel):

    def __init__(
            self,
            nFilter,
            _KERNEL_SIZE,
            CHANNEL_FACTOR1,
            CHANNEL_FACTOR2,
            nlayers = 8,
            dropout = 0,
            output_hidden_size = None,
            **kwargs
        ):
        super(ResGSModel_LCL, self).__init__(**kwargs)

        if torch.cuda.is_available():
            # Due to the structure of the training function, the input tensor are on cuda:0 and the output tensor should be on cuda:0
            self.IOdevice = "cuda:0"
            if torch.cuda.device_count() > 1:
                # Enable to use a second gpu for the training of the model.
                self.computeDevice = "cuda:1"
            else:
                self.computeDevice = "cuda:0"
        else:
            self.IOdevice = "cpu"
            self.computeDevice = "cpu"
        
        self.input_block1 = Res_Block(1, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1).to(self.IOdevice)
        self.input_block2 = Res_Block(nFilter, nb_filter=nFilter, kernel_size=_KERNEL_SIZE, strides=1).to(self.IOdevice)
        nFilter1 = int(nFilter * CHANNEL_FACTOR1)

        self.layer1 = nn.Sequential(
            *[nn.Sequential( 
                LocalLinear(math.ceil(36304 / (2**(i-1))), nFilter1, kernel_size=_KERNEL_SIZE, stride=2, in_channels= int(nFilter * CHANNEL_FACTOR2**(i-1))),
                nn.ReLU(),
                nn.BatchNorm1d(nFilter1),
                Conv1d_BN(nFilter1, nb_filter=int(nFilter * CHANNEL_FACTOR2**i), kernel_size=1, strides=1, padding=0), 
                Res_Block(int(nFilter * CHANNEL_FACTOR2**i), nb_filter=int(nFilter * CHANNEL_FACTOR2**i), kernel_size=_KERNEL_SIZE, strides=1), 
                Res_Block(int(nFilter * CHANNEL_FACTOR2**i), nb_filter=int(nFilter * CHANNEL_FACTOR2**i), kernel_size=_KERNEL_SIZE, strides=1),
            )for i in range(1, max(2, nlayers - 11)) ]).to(self.computeDevice)
        
        self.layers2 = nn.Sequential(
            *[nn.Sequential( 
                LocalLinear(math.ceil(36304 / (2**(i-1))), nFilter1, kernel_size=_KERNEL_SIZE, stride=2, in_channels= int(nFilter * CHANNEL_FACTOR2**(i-1))),
                nn.ReLU(),
                nn.BatchNorm1d(nFilter1),
                Conv1d_BN(nFilter1, nb_filter=int(nFilter * CHANNEL_FACTOR2**i), kernel_size=1, strides=1, padding=0), 
                Res_Block(int(nFilter * CHANNEL_FACTOR2**i), nb_filter=int(nFilter * CHANNEL_FACTOR2**i), kernel_size=_KERNEL_SIZE, strides=1), 
                Res_Block(int(nFilter * CHANNEL_FACTOR2**i), nb_filter=int(nFilter * CHANNEL_FACTOR2**i), kernel_size=_KERNEL_SIZE, strides=1),
            )for i in range(max(2, nlayers - 11), nlayers + 1) ]).to(self.IOdevice)

        # Size of the penultimate dimention of the tensor after all residual blocks: each layer divide this dimention by two
        n_elements = math.ceil(36304 / (2**nlayers))

        # Size of the channels (ie last dimention of the tensor after all residual blocks)
        n_channels = int(nFilter * CHANNEL_FACTOR2**nlayers)

        # Determine the number of channels to have an input dimention close to 6400 in the linear layer
        filter_near_6400 = 6400 // n_elements
        if filter_near_6400 == 0:
            filter_near_6400 = 1
            
        if output_hidden_size == None or output_hidden_size <= 1:
            self.output = nn.Sequential(
                Conv1d_BN(n_channels, nb_filter= filter_near_6400, kernel_size=1, strides=1, padding=0),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(filter_near_6400 * n_elements, 1)
            ).to(self.IOdevice)
        else:
            self.output = nn.Sequential(
                Conv1d_BN(n_channels, nb_filter= filter_near_6400, kernel_size=1, strides=1, padding=0),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(filter_near_6400 * n_elements, output_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_hidden_size, 1)
            ).to(self.IOdevice)

    def forward(self, x):
        # Set the number of channels to 1 as required by the conv1d layer
        x = x.view(x.shape[0], 1, x.shape[1])
        
        x = self.input_block1(x)
        x = self.input_block2(x)

        x = self.layer1(x.to(self.computeDevice))
        x = self.layers2(x.to(self.IOdevice))

        x = self.output(x)
        return x