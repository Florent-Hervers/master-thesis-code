import torch

from torch import nn
from enum import Enum

class EmbeddingType(Enum):
    Linear = 1
    EmbeddingTable = 2

class TransformerBlock(nn.Module):
    def __init__(self,embedding_size, n_hidden, n_heads, dropout = 0):
        super(TransformerBlock, self).__init__()

        self.multihead = nn.MultiheadAttention(embedding_size, n_heads, batch_first=True, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(embedding_size, n_hidden)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(n_hidden, embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
    
    def forward(self, x):
        y, _ = self.multihead(x,x,x, need_weights=False)
        y = self.norm1(x + y)
        z = self.fc1(self.dropout1(y))
        z = self.fc2(self.dropout2(self.relu(z)))
        return self.norm2(y + z)
    
class GPTransformer(nn.Module):
    def __init__(self,
                 n_features,
                 embedding_size,
                 n_hidden,
                 n_heads,
                 n_blocks,
                 sequence_length,
                 dropout = 0,
                 output_hidden_size = None,
                 embedding_type: EmbeddingType = EmbeddingType.Linear,
                 embedding_table_weight = None):
        """Create the GPTransformer model with the given argument

        Args:
            n_features (int): number of possible values that the input can take. Ignored when embedding_type is equal to EmbeddingType.Linear or if embedding_table_weight is provided (the n_features will be infered from the tensor shape).
            embedding_size (int): size of the embeddding of the markers.
            n_hidden (int): hidden size of the feedforward block
            n_heads (int): number of heads in the multi-head attention layers
            n_blocks (int): number of transformer blocks (attention + feed-forward) of the model
            sequence_length (int): length of the sequence fed as input.
            dropout (int ,optional): Probability for all the dropout layer of the model. Defaults to 0.
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
            
        if embedding_type == EmbeddingType.Linear:
            self.embedding = nn.Linear(sequence_length, sequence_length * embedding_size).to(self.IOdevice)
            
            # Resize the vector as the Linear layer provide the vector flattened.
            self.preprocessing = lambda x: x.view((x.shape[0], sequence_length, embedding_size))
        elif embedding_type == EmbeddingType.EmbeddingTable:
            if embedding_table_weight != None:
                self.embedding = nn.Embedding.from_pretrained(embedding_table_weight).to(self.IOdevice)
            else:
                self.embedding = nn.Embedding(n_features, embedding_size).to(self.IOdevice)
            
            # Add positionnal encoding
            self.preprocessing = PositionalEncoding(embedding_size, max_len=36304).to(self.IOdevice)

        self.transformer = nn.Sequential(
            *[TransformerBlock(embedding_size, n_hidden, n_heads, dropout) for _ in range(n_blocks)]
        ).to(self.computeDevice)

        if output_hidden_size == None or output_hidden_size <= 1:
            self.output = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embedding_size * sequence_length, 1),
            ).to(self.computeDevice)
        else:
            self.output = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embedding_size * sequence_length, output_hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_hidden_size, 1)
            ).to(self.computeDevice)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.preprocessing(x)
        x = self.transformer(x.to(self.computeDevice))
        return self.output(x.view(x.shape[0], -1)).to(self.IOdevice)
    

class PositionalEncoding(nn.Module):
    """ Implementation based on the sinusoidal positionnal encoding provided by d2l (source: https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html)
    """
    def __init__(self, d_model: int, dropout: float = 0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        if torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

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
        X = X + self.P[:, :X.shape[1], :self.d_model].to(self.device)
        return self.dropout(X)