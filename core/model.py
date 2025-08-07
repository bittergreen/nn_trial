import torch
import torch.nn as nn
import torch.nn.functional as F
import math

n_layer = 6
n_head = 6
head_size = 16
n_embd = 64
dropout = 0.2
batch_size = 64
sequence_length = 256
temperature = 1.0
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class LoRALinear(nn.Module):
    def __init__(self, linear, rank=4, lora_alpha=16, lora_dropout=0.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.rank
        self.dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()
        self.lora_A = nn.Parameter(torch.empty(linear.in_features, self.rank))
        self.lora_B = nn.Parameter(torch.empty(self.rank, linear.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.linear(x) + (self.dropout(x) @ self.lora_A @ self.lora_B) * self.scaling


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads: int, head_size: int, use_lora: bool = False, lora_rank: int = 4, lora_alpha: int = 16, lora_dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        q_linear = nn.Linear(n_embd, head_size * num_heads, device=device)
        self.query = LoRALinear(q_linear, lora_rank, lora_alpha, lora_dropout) if use_lora else q_linear
        k_linear = nn.Linear(n_embd, head_size * num_heads, device=device)
        self.key = LoRALinear(k_linear, lora_rank, lora_alpha, lora_dropout) if use_lora else k_linear
        v_linear = nn.Linear(n_embd, head_size * num_heads, device=device)
        self.value = LoRALinear(v_linear, lora_rank, lora_alpha, lora_dropout) if use_lora else v_linear
        out_linear = nn.Linear(head_size * num_heads, n_embd, device=device)
        self.output = LoRALinear(out_linear, lora_rank, lora_alpha, lora_dropout) if use_lora else out_linear
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

    def __init__(self, n_embd: int, use_lora: bool = False, lora_rank: int = 4, lora_alpha: int = 16, lora_dropout: float = 0.0):
        super().__init__()
        fc1 = nn.Linear(n_embd, 4 * n_embd, device=device)
        fc2 = nn.Linear(4 * n_embd, n_embd, device=device)
        self.net = nn.Sequential(
            LoRALinear(fc1, lora_rank, lora_alpha, lora_dropout) if use_lora else fc1,
            nn.ReLU(),
            LoRALinear(fc2, lora_rank, lora_alpha, lora_dropout) if use_lora else fc2,
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd: int, n_head: int, head_size: int, use_lora: bool = False, lora_rank: int = 4, lora_alpha: int = 16, lora_dropout: float = 0.0):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, head_size, use_lora, lora_rank, lora_alpha, lora_dropout)
        self.ffn = FeedForward(n_embd, use_lora, lora_rank, lora_alpha, lora_dropout)
        self.ln1 = nn.LayerNorm(n_embd, device=device)
        self.ln2 = nn.LayerNorm(n_embd, device=device)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class MiniGPT(nn.Module):

    def __init__(self, vocab_size: int, n_layer: int = n_layer, n_head: int = n_head, head_size: int = head_size, use_lora: bool = False, lora_rank: int = 4, lora_alpha: int = 16, lora_dropout: float = 0.0):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, device=device)
        self.position_embedding_table = nn.Embedding(sequence_length, n_embd, device=device)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, head_size, use_lora, lora_rank, lora_alpha, lora_dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd, device=device)
        lm_head_linear = nn.Linear(n_embd, vocab_size, device=device)
        self.lm_head = LoRALinear(lm_head_linear, lora_rank, lora_alpha, lora_dropout) if use_lora else lm_head_linear

    @classmethod
    def from_pretrained(cls, checkpoint_path: str, use_lora: bool = False, **kwargs):
        # Create a plain model without LoRA
        plain_model = cls(**kwargs, use_lora=False)
        plain_model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        if not use_lora:
            return plain_model

        # Create a LoRA model
        lora_model = cls(**kwargs, use_lora=True)

        # Copy weights from plain_model to lora_model's inner linears
        lora_model.token_embedding_table.load_state_dict(plain_model.token_embedding_table.state_dict())
        lora_model.position_embedding_table.load_state_dict(plain_model.position_embedding_table.state_dict())
        lora_model.ln_f.load_state_dict(plain_model.ln_f.state_dict())

        for p_block, l_block in zip(plain_model.blocks, lora_model.blocks):
            l_block.ln1.load_state_dict(p_block.ln1.state_dict())
            l_block.ln2.load_state_dict(p_block.ln2.state_dict())
            l_block.sa.query.linear.load_state_dict(p_block.sa.query.state_dict())
            l_block.sa.key.linear.load_state_dict(p_block.sa.key.state_dict())
            l_block.sa.value.linear.load_state_dict(p_block.sa.value.state_dict())
            l_block.sa.output.linear.load_state_dict(p_block.sa.output.state_dict())
            # Copy mask buffer
            l_block.sa.register_buffer("mask", p_block.sa.mask)
            l_block.ffn.net[0].linear.load_state_dict(p_block.ffn.net[0].state_dict())
            l_block.ffn.net[2].linear.load_state_dict(p_block.ffn.net[2].state_dict())

        lora_model.lm_head.linear.load_state_dict(plain_model.lm_head.state_dict())

        return lora_model
    
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

