import torch
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class HippocampalMemorySystem(nn.Module):

    """
    """

    def __init__(self, key_dim: int, context_dim: int):
        super().__init__()
        self.encoder = nn.Linear(key_dim, key_dim, device=device)
        # trainable representation of inner contexts
        self.context_repr = nn.Embedding(key_dim, context_dim, device=device)
        self.decoder = nn.Linear(context_dim, key_dim, device=device)
    
    def encode(self, x):
        # encoding input x and update inner context representation
        related_context = self.retrieve(x)
        self.context_repr.weight[related_context] = self.encoder(x)
        pass
    
    def retrieve(self, x):
        return self.context_repr(x)
    
    def forward(self, x):
        self.encode(x)
        out = self.retrieve(x)
        return out


class Hippocampus(nn.Module):

    """
    Binds contexts of different modalities into a single representation.
    Indexed by a sparse key.
    """
    def __init__(self, input_dim: int, key_dim: int, context_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.key_dim = key_dim
        # Todo: should be combining multiple modalities into a single index
        self.dentate_gyrus = nn.Linear(input_dim, key_dim, device=device)  # generate sparse index
        self.memory = HippocampalMemorySystem(key_dim, context_dim)
    
    def forward(self, x):
        index = self.dentate_gyrus(x)
        sparsity_loss = 1e-4 * torch.mean(torch.abs(index))  # enforce sparsity
        inner_context = self.memory(index)
        return inner_context, sparsity_loss
    
    def replay(self, gen_length: int):
        pass
