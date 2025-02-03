from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, n_hidden, n_heads):
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
    def __init__(self,  n_features, embedding_size, n_hidden, n_heads, n_blocks):
        super(GPTransformer, self).__init__()
        self.n_features = n_features
        self.embedding_size = embedding_size
        self.embedding = nn.Linear(n_features, n_features * embedding_size) #nn.Embedding(3, embedding_size)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embedding_size, n_hidden, n_heads) for _ in range(n_blocks)]
        )
        self.output = nn.Linear(embedding_size * n_features, 1)
    
    def forward(self, x):
        x = self.embedding(x) #(x.int())
        x = x.view((x.shape[0], self.n_features, self.embedding_size))
        x = self.transformer(x)
        return self.output(x.view(x.shape[0], -1))