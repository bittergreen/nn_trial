import torch
import torch.nn as nn
import torch.nn.functional as F
import math


n_layer = 4
n_head = 8
head_size = 16
n_embd = 8
dropout = 0.1
batch_size = 1
sequence_length = 128
temperature = 1.0
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

context_dim = 128


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size * num_heads, device=device)
        self.key = nn.Linear(n_embd, head_size * num_heads, device=device)
        self.value = nn.Linear(n_embd, head_size * num_heads, device=device)
        self.output = nn.Linear(head_size * num_heads, n_embd, device=device)
        self.register_buffer("mask", torch.tril(torch.ones(sequence_length, sequence_length, device=device)))
        self.dropout = nn.Dropout(dropout)
        self.scale = head_size ** 0.5
    
    def forward(self, x):
        # x -> (B, T, C), B = batch size, T = sequence length, C = embedding dimension
        B, T, C = x.shape

        q = self.query(x)  # (B, T, H)
        k = self.key(x)  # (B, T, H)
        v = self.value(x)  # (B, T, H)

        qk = q @ k.transpose(-2, -1) * self.scale  # (B, T, T)
        qk = qk.masked_fill(self.mask[:T, :T] == 0, float("-inf"))  # (B, T, T)
        qk = qk.softmax(dim=-1)  # (B, T, T)
        qk = self.dropout(qk)  # (B, T, T)

        out = qk @ v  # (B, T, H)
        out = self.output(out)  # (B, T, C)
        return out
        

class FeedForward(nn.Module):

    def __init__(self, n_embd: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, device=device),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd, device=device),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd: int, n_head: int, head_size: int, use_cross: bool = False):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffn = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd, device=device)
        self.ln2 = nn.LayerNorm(n_embd, device=device)
        self.use_cross = use_cross
        if use_cross:
            self.cross = CrossMultiHeadAttention(n_head, head_size)
            self.ln3 = nn.LayerNorm(n_embd, device=device)
    
    def forward(self, x, memory=None):
        y = self.sa(self.ln1(x))
        if self.use_cross:
            y = y + self.cross(self.ln3(x + y), memory)
        x = x + y
        x = x + self.ffn(self.ln2(x))
        return x


class CrossMultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, head_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size * num_heads, device=device)
        self.key = nn.Linear(n_embd, head_size * num_heads, device=device)
        self.value = nn.Linear(n_embd, head_size * num_heads, device=device)
        self.output = nn.Linear(head_size * num_heads, n_embd, device=device)
        self.dropout = nn.Dropout(dropout)
        self.scale = head_size ** -0.5

    def forward(self, x, memory):
        B, T, C = x.shape
        q = self.query(x)  # (B, T, H)
        k = self.key(memory)  # (B, W, H)
        v = self.value(memory)  # (B, W, H)
        qk = q @ k.transpose(-2, -1) * self.scale  # (B, T, W)
        qk = qk.softmax(dim=-1)
        qk = self.dropout(qk)
        out = qk @ v  # (B, T, H)
        out = self.output(out)
        return out


class ContextMerger(nn.Module):
    def __init__(self, input_dim, context_dim):
        super().__init__()
        self.projection = nn.Linear(input_dim, context_dim, device=device)

    def forward(self, x):
        # x: (B, T, input_dim)
        projected = self.projection(x)
        # additively merge previous information of the masked tokens into a new (B, T, context_dim)
        merged = torch.cumsum(projected, dim=1)
        return projected


class NeuralMemtable(nn.Module):

    """
    This is the hippocampal representation of inner context, 
    to which we can apply cross-attention in late transformer layers
    """
    def __init__(self, n_embd: int, num: int = 8, temperature: float = 0.1, alpha=0.2):
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
        similarity = torch.einsum('btd,nd->btn', q, k_memory)  # (B, T, num)
        
        # Apply softmax to get segment weights
        attention = torch.softmax(similarity / self.temperature, dim=-1)  # (B, T, num)

        retrieved_context = torch.einsum('btn,nd->btd', attention, self.memory)  # (B, T, n_embd)

        return retrieved_context


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
        self.out_proj = nn.Linear(context_dim, n_embd, device=device)
    
    def forward(self, x):
        index = self.dentate_gyrus(x)
        out = self.memtable(index)  # (B, T, context_dim)
        out = self.out_proj(out)
        return out


class MiniGPT(nn.Module):

    def __init__(self, vocab_size: int, n_layer: int = n_layer, n_head: int = n_head, head_size: int = head_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, device=device)
        self.position_embedding_table = nn.Embedding(sequence_length, n_embd, device=device)
        self.early_blocks = nn.Sequential(*[Block(n_embd, n_head, head_size) for _ in range(n_layer-1)])
        self.hippo = Hippocampus(n_embd, 4 * n_embd)
        self.final_block = Block(n_embd, n_head, head_size, use_cross=True)
        self.ln_f = nn.LayerNorm(n_embd, device=device)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, switch=None, cue=None):
        B, T = idx.shape
        emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = emb + pos_emb
        x = self.early_blocks(x)
        past_context = self.hippo(x)
        x = self.final_block(x, past_context)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens, switch=None, cue=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -sequence_length:]
            logits, _, _ = self(idx_cond, switch=switch, cue=cue)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
        
    def interactive_prompt(self, encode, decode):
        while True:
            prompt = input("\nEnter a prompt (or 'exit' to quit): ")
            if prompt.lower() == 'exit':
                break
                
            context = torch.tensor(encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
            generated = self.generate(context, max_new_tokens=500)
            generated_text = decode(generated[0].tolist())
            print(f"\nGenerated text:\n{generated_text}")


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_interval):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            if split == "train":
                X, Y = train_data.get_batch(batch_size=batch_size, block_size=sequence_length)
            else:
                X, Y = val_data.get_batch(batch_size=batch_size, block_size=sequence_length)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out


def train_model(model, train_data, val_data, lr, max_iters, eval_interval):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, eval_interval)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        X, Y = train_data.get_batch(batch_size=batch_size, block_size=sequence_length)
        logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward() 
        optimizer.step()

