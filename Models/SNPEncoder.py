import torch
import torch.nn as nn

from Models.LCLNN import LocalLinear

class SNPEncoder(nn.Module):
    supported_modes = ["convolution", "LCL"]
    def __init__(self, mode):
        super(SNPEncoder, self).__init__()

        if mode not in self.supported_modes:
            raise Exception(f"The mode {mode} isn't supported. Supported modes are {self.supported_modes}")
        
        self.one_hot = nn.Embedding.from_pretrained(torch.eye(3))
        self.flatten = nn.Flatten()
        if mode == "convolution":
            self.encoder = nn.Conv1d(1, 1, 3, stride=3)
        elif mode == "LCL":
            self.encoder = LocalLinear(36304 * 3, 1, 3, 3)
    
    def forward(self, x):
        x = self.one_hot(x.type(torch.int32))

        x = self.flatten(x)

        if len(x.shape) == 2:
            x = x.view(x.shape[0], 1, x.shape[1])
        
        if len(x.shape) != 3:
            raise Exception(f"Bad dimention for the input, only 2 or 3 dimentional input are allowed but got {len(x.shape)} instead!")

        x = self.encoder(x)

        # Remove the channel dimention to be able to use the encoder with non convolutionnal networks
        return x.view(x.shape[0], x.shape[-1])   