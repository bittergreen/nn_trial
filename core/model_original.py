import torch
import torch.nn as nn
import torch.nn.functional as F

n_layer = 4
n_head = 8
head_size = 16
n_embd = 8
dropout = 0.1
batch_size = 32
sequence_length = 128
temperature = 1.0
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


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

    def __init__(self, n_embd: int, n_head: int, head_size: int):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffn = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd, device=device)
        self.ln2 = nn.LayerNorm(n_embd, device=device)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):

    def __init__(self, vocab_size: int, n_layer: int = n_layer, n_head: int = n_head, head_size: int = head_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, device=device)
        self.position_embedding_table = nn.Embedding(sequence_length, n_embd, device=device)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, head_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, device=device)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = emb + pos_emb
        x = self.blocks(x)
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
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -sequence_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


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

