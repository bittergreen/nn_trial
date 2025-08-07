import torch
from torch.cuda import temperature
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


max_context_num = 4
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

"""
[block_output]
     |
block output is the information carried by each token, shape (B, T, C)

We should additively merge the masked T into a new (B, T, context_dim), 
where each tensor in position t encode everything before it in the sequence and itself.
"""


class ContextMerger(nn.Module):
    def __init__(self, input_dim, context_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, context_dim, device=device)

    def forward(self, x):
        # x: (B, T, input_dim)
        projected = self.projection(x)
        # additively merge previous information of the masked tokens into a new (B, T, context_dim)
        merged = torch.cumsum(projected, dim=-1)
        return merged


class NeuralMemtable(nn.Module):

    """
    This is the hippocampal representation of inner context, 
    to which we can apply cross-attention in late transformer layers
    """
    def __init__(self, n_embd: int, num: int = 4, temperature: float = 0.1, alpha=0.2):
        super().__init__()
        self.dim = n_embd
        self.num = num
        self.alpha = alpha
        self.temperature = temperature
        # Todo: qks here can be smaller in size? To reduce computation
        self.to_q = nn.Linear(n_embd, n_embd, device=device)
        self.to_k = nn.Linear(n_embd, n_embd, device=device)
        # memory serves as the value
        memory_tensor = torch.randn(num, n_embd, device=device)
        self.memory = nn.Parameter(torch.nn.functional.normalize(memory_tensor, dim=-1))  # (num, n_embd)
    
    def forward(self, x):
        # x -> (B, T, n_embd)
        q = self.to_q(x)  # (B, T, n_embd)
        k_memory = self.to_k(self.memory)  # (num, n_embd)
        
        # Calculate attention scores using einsum
        similarity = torch.einsum('btd,nd->btn', q, k_memory)  # (B, T, num)
        
        # Apply softmax to get segment weights
        attention = torch.softmax(similarity / self.temperature, dim=-1)  # (B, T, num)
        
        # Select memory segment with attention weights using einsum
        retrieved_context = torch.einsum('btn,nd->btd', attention, self.memory)  # (B, T, n_embd)

        # Update memory using einsum for the last timestep
        # First compute the weighted input for the last timestep
        last_step_update = torch.einsum('btn,btd->nd', 
                                       attention[:,-1:,:], 
                                       x[:,-1:,:]).mean(dim=0)  # (num, n_embd)
        
        self.memory = self.memory + self.alpha * last_step_update

        return retrieved_context.squeeze(1)


class Hippocampus(nn.Module):

    def __init__(self, input_dim: int, context_dim: int):
        # input_dim: original C in the (B, T, C)
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        # Todo: should be combining multiple modalities into a single index
        # Map (B, T, C) to (B, T, context_dim), context_dim larger than C but smaller than T*C
        self.dentate_gyrus = ContextMerger(input_dim, context_dim)  # generate sparse index
        self.memtable = NeuralMemtable(context_dim)
    
    def forward(self, x):
        index = self.dentate_gyrus(x)
        out = self.memtable(index)
        return out


