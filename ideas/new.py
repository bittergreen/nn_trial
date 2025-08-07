import torch
import torch.nn as nn

class SegmentedMemoryModule(nn.Module):
    def __init__(self, feature_dim, num_segments=4, temperature=0.1):
        """
        Initialize segmented memory module
        feature_dim: dimension of each memory slot
        num_segments: number of memory segments (default 4)
        temperature: temperature for softmax (controls sharpness)
        """
        super().__init__()
        self.num_segments = num_segments
        # Initialize memory as a learnable parameter and normalize it
        memory_tensor = torch.randn(num_segments, feature_dim)
        self.memory = nn.Parameter(torch.nn.functional.normalize(memory_tensor, dim=-1))
        self.temperature = temperature
    
    def forward(self, x):
        """
        x: input tensor of shape [batch_size, feature_dim]
        Returns the most similar memory segment
        """
        # Normalize input and memory
        x_normalized = torch.nn.functional.normalize(x, dim=-1)
        
        # Compute cosine similarity between input and all memory segments
        similarity = torch.matmul(x_normalized, self.memory.T)  # [batch_size, num_segments]
        
        # Apply softmax to get segment weights
        attention = torch.softmax(similarity / self.temperature, dim=-1)  # [batch_size, num_segments]
        
        # Select memory segment with highest attention (still differentiable due to softmax)
        output = torch.matmul(attention.unsqueeze(1), self.memory)  # [batch_size, 1, feature_dim]
        
        return output.squeeze(1)
