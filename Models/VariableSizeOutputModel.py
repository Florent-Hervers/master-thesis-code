import torch.nn as nn

class VariableSizeOutputModel(nn.Module):
    output_size : int = 1
    def __init__(self, **kwargs):
        super(VariableSizeOutputModel, self).__init__()
        if "output_size" in kwargs.keys():
            self.output_size = kwargs["output_size"]