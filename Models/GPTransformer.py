import torch

from torch import nn
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
        y, _ = self.multihead(x,x,x, need_weights=False)
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
                 output_hidden_size = None,
                 embedding_type: EmbeddingType = EmbeddingType.Linear,
                 embedding_table_weight = None):
        """Create the GPTransformer model with the given argument

        Args:
            n_features (int): Number of markers selected.
            embedding_size (int): size of the embeddding of the marker
            n_hidden (int): hidden size of the feedforward block
            n_heads (int): number of heads in the multi-head attention layers
            n_blocks (int): number of transformer blocks (attention + feed-forward) of the model
            output_hidden_size (int, optional): Size of the hidden layer of the output mlp. Defaults to None (only one linear layer)
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

        if output_hidden_size == None or output_hidden_size <= 1:
            self.output = nn.Linear(embedding_size * n_features, 1)
        else:
            self.output = nn.Sequential(
                nn.Linear(embedding_size * n_features, output_hidden_size),
                nn.ReLU(),
                nn.Linear(output_hidden_size, 1)
            )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.preprocessing(x)
        x = self.transformer(x)
        return self.output(x.view(x.shape[0], -1))
    

class PositionalEncoding(nn.Module):
    """ Implementation based on the sinusoidal positionnal encoding provided by d2l (source: https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html)
    """
    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Store the true embedding size to restore it at run time
        self.d_model = d_model
        
        # Extend the argument to manage odd embedding size (which requires an extra value for the sin but not for the cos)
        if d_model % 2 == 1:
            d_model += 1

        self.P = torch.zeros((1, max_len, d_model))
        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, d_model, 2, dtype=torch.float32) / d_model)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :self.d_model].to(X.device)
        return self.dropout(X)