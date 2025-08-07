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


class HippocampalContextRepresentation(nn.Module):

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
        self.context_repr = HippocampalContextRepresentation(key_dim, context_dim)
    
    def forward(self, x):
        index = self.dentate_gyrus(x)
        context = self.context_repr(index)
        return context
    
    def generate(self, gen_length: int):
        pass


class HippocampalModule(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, sparse_dim: int, sparse_k: int, window_size: int, max_contexts: int = 16, sim_threshold: float = 0.8):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sparse_dim = sparse_dim
        self.sparse_k = sparse_k
        self.window_size = window_size
        self.sim_threshold = sim_threshold
        self.max_contexts = max_contexts
        self.encoder = nn.LSTM(input_dim, latent_dim, batch_first=True, device=device)
        self.sparse_encoder = nn.Linear(latent_dim, sparse_dim, device=device)
        self.gen_gru = nn.GRU(input_dim, input_dim, batch_first=True, device=device)
        self.start_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.memory_keys = nn.Parameter(torch.randn(max_contexts, sparse_dim, device=device))
        self.memory_values = nn.Parameter(torch.randn(max_contexts, latent_dim, device=device))
        self.register_buffer('memory_usage', torch.zeros(max_contexts, device=device))
        self.register_buffer('current_key', torch.zeros(batch_size, sparse_dim, device=device))
        self.register_buffer('current_value', torch.zeros(batch_size, latent_dim, device=device))
        self.register_buffer('is_initialized', torch.zeros(batch_size, dtype=torch.bool, device=device))

    def reset_state(self):
        self.current_key.zero_()
        self.current_value.zero_()
        self.is_initialized.fill_(False)

    def make_sparse(self, index):
        values, indices = torch.topk(index.abs(), self.sparse_k, dim=-1)
        sign = torch.sign(index.gather(-1, indices))
        sparse = torch.zeros_like(index).scatter_(-1, indices, values * sign)
        return sparse

    def forward(self, x, switch=None, cue=None):
        if switch is None:
            switch = torch.zeros(x.shape[0], dtype=torch.bool, device=device)
        B, T, C = x.shape
        latent, (h, c) = self.encoder(x)
        new_rep = h.squeeze(0)  # (B, latent_dim)
        sparse_index = self.make_sparse(self.sparse_encoder(new_rep))  # (B, sparse_dim)

        mask_init = ~self.is_initialized
        self.current_key[mask_init] = sparse_index[mask_init]
        self.current_value[mask_init] = new_rep[mask_init]
        self.is_initialized[mask_init] = True

        mask_update = self.is_initialized & ~switch
        self.current_value[mask_update] = 0.9 * self.current_value[mask_update] + 0.1 * new_rep[mask_update]

        if switch.any():
            if cue is None:
                raise ValueError("Cue required for switch")
            if cue.dim() == 2:
                cue = cue.unsqueeze(1)  # Assume T_c=1 if vector
            if cue.shape[0] != B:
                raise ValueError("Cue batch size mismatch")
            _, (h_c, _) = self.encoder(cue)
            cue_rep = h_c.squeeze(0)  # (B, latent_dim)
            cue_sparse = self.make_sparse(self.sparse_encoder(cue_rep))  # (B, sparse_dim)
            for b in torch.nonzero(switch).squeeze(1):
                sim_b = F.cosine_similarity(cue_sparse[b:b+1], self.memory_keys, dim=-1)
                max_sim_b, max_id_b = sim_b.max(-1)
                if max_sim_b.item() > self.sim_threshold:
                    self.current_key[b] = self.memory_keys[max_id_b]
                    self.current_value[b] = self.memory_values[max_id_b]
                    self.memory_usage[max_id_b] += 1
                else:
                    slot = self.memory_usage.argmin().item()
                    self.memory_keys.data[slot] = cue_sparse[b]
                    self.memory_values.data[slot] = cue_rep[b]
                    self.memory_usage[slot] += 1
                    self.current_key[b] = self.memory_keys[slot]
                    self.current_value[b] = self.memory_values[slot]
                self.is_initialized[b] = True

        start_token = self.start_token.repeat(B, 1, 1)
        hidden = self.current_value.unsqueeze(0)  # (1, B, latent_dim)
        input_t = start_token
        window = []
        for _ in range(self.window_size):
            output, hidden = self.gen_gru(input_t, hidden)
            window.append(output)
            input_t = output
        window = torch.cat(window, dim=1)  # (B, window_size, input_dim)
        return window


class MiniGPT(nn.Module):

    def __init__(self, vocab_size: int, n_layer: int = n_layer, n_head: int = n_head, head_size: int = head_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, device=device)
        self.position_embedding_table = nn.Embedding(sequence_length, n_embd, device=device)
        self.early_blocks = nn.Sequential(*[Block(n_embd, n_head, head_size) for _ in range(n_layer-1)])
        self.hippo = HippocampalModule(n_embd, n_embd, 64, 8, 16)
        self.final_block = Block(n_embd, n_head, head_size, use_cross=True)
        self.ln_f = nn.LayerNorm(n_embd, device=device)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, switch=None, cue=None):
        B, T = idx.shape
        emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = emb + pos_emb
        x = self.early_blocks(x)
        window = self.hippo(x, switch=switch, cue=cue)
        x = self.final_block(x, window)
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
            logits, _ = self(idx_cond, switch=switch, cue=cue)
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
            model.hippo.reset_state()
            logits, loss = model(X, Y, switch=None, cue=None)
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
        model.hippo.reset_state()
        logits, loss = model(X, Y, switch=None, cue=None)
        optimizer.zero_grad(set_to_none=True)
        loss.backward() 
        optimizer.step()

