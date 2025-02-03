import torch
from torch import nn
import torch.nn.functional as F

class MLP(torch.nn.Module):
    def __init__(self, nlayers: int = 1, hidden_nodes: list[int] = [], dropout: float = 0):
        super(MLP, self).__init__()
        
        if dropout < 0 or dropout >= 1:
            raise AttributeError("The dropout must be between 0 and 1")

        if nlayers < 1:
            raise AttributeError("The number of layers must be greater or equal than one !")
        
        if len(hidden_nodes) != nlayers - 1:
            raise AttributeError(f"Not enough hidden_nodes given, expected a list of length {nlayers - 1} but got one of {len(hidden_nodes)}")

        # Use a copy to avoid modifying the hyperparameter value for future runs
        hidden_nodes_model = hidden_nodes.copy()
        hidden_nodes_model.insert(0, 36304)
        hidden_nodes_model.append(1)

        self.model = nn.Sequential(*[LinearBlock(hidden_nodes_model[i], hidden_nodes_model[i + 1], dropout=dropout) for i in range(nlayers - 1)])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_nodes_model[-2], hidden_nodes_model[-1])

    def forward(self, x):
        return self.output_layer(self.dropout(self.model(x)))

class LinearBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout = 0):
        super(LinearBlock, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features=input_size, out_features=output_size)
    
    def forward(self,x):
        return F.relu(self.fc(x))