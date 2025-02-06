import torch
import math

from torch import nn, Tensor
from enum import Enum

class EmbeddingType(Enum):
    Linear = 1
    EmbeddingTable = 2

class TransformerBlock(nn.Module):
    def __init__(self,embedding_size, n_hidden, n_heads):
        super(TransformerBlock, self).__init__()

        self.multihead = nn.MultiheadAttention(embedding_size, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.fc1 = nn.Linear(embedding_size, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
    
    def forward(self, x):
        y, _ = self.multihead(x,x,x)
        y = self.norm1(x + y)
        z = self.fc1(y)
        z = self.fc2(self.relu(z))
        return self.norm2(y + z)
    
class GPTransformer(nn.Module):
    def __init__(self,
                 n_features,
                 embedding_size,
                 n_hidden,
                 n_heads,
                 n_blocks,
                 embedding_type: EmbeddingType = EmbeddingType.Linear,
                 embedding_table_weight = None):
        """Create the GPTransformer model with the given argument

        Args:
            n_features (int): Number of markers selected.
            embedding_size (int): size of the embeddding of the marker
            n_hidden (int): hidden size of the feedforward block
            n_heads (int): number of heads in the multi-head attention layers
            n_blocks (int): number of transformer blocks (attention + feed-forward) of the model
            embedding_type (EmbeddingType, optional): Type of the embedding to use. EmbeddingType.Linear will use a linear layer to construct the embeddings. \ 
            EmbeddingType.EmbeddingTable will use an embedding table with a sinusoidal positionnal encoding. Defaults to EmbeddingType.Linear.
            embedding_table_weight (_type_, optional): If embedding_type is EmbeddingType.EmbeddingTable, the embeddings weigths can be provided if wanted. Defaults to None.

        Raises:
            ValueError: if the argument embedding_type isn't of type EmbeddingType
        """
        if type(embedding_type) != EmbeddingType:
            raise ValueError(f"The type of argument embedding_type should be the enum EmbeddingType but got {type(embedding_type)} instead!")
        
        super(GPTransformer, self).__init__()
 
        if embedding_type == EmbeddingType.Linear:
            self.embedding = nn.Linear(n_features, n_features * embedding_size)
            
            # Resize the vector as the Linear layer provide the vector flattened.
            self.preprocessing = lambda x: x.view((x.shape[0], n_features, embedding_size))
        elif embedding_type == EmbeddingType.EmbeddingTable:
            if embedding_table_weight != None:
                self.embedding = nn.Embedding.from_pretrained(embedding_table_weight)
            else:
                self.embedding = nn.Embedding(3, embedding_size)
            
            # Add positionnal encoding
            self.preprocessing = PositionalEncoding(embedding_size, max_len=36304)

        self.transformer = nn.Sequential(
            *[TransformerBlock(embedding_size, n_hidden, n_heads) for _ in range(n_blocks)]
        )
        self.output = nn.Linear(embedding_size * n_features, 1)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.preprocessing(x)
        x = self.transformer(x)
        return self.output(x.view(x.shape[0], -1))
    

class PositionalEncoding(nn.Module):
    """ Implementation of the sinusoidal positionnal encoding provided by a depreciated pytorch tutorial (source: https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch)
    """
    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        # The torch.arrange(2, d_model, 2) create the 2k of the mathematical formulation
        div_term = torch.exp(torch.arange(2, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)