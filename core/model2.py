import torch
import torch.nn as nn
import torch.nn.functional as F

n_layer = 6
n_head = 6
head_size = 16
n_embd = 64
dropout = 0.2
batch_size = 64
sequence_length = 256
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


class HippocampalEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, latent_dim, num_layers=1, batch_first=True, device=device)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: (B, T, C) -> Process sequence with LSTM
        latent, _ = self.lstm(x)  # (B, T, latent_dim)
        latent = self.activation(latent)
        return latent


class HippocampalDecoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, output_dim, num_layers=1, batch_first=True, device=device)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: (B, T, latent_dim) -> Process to reconstruct sequence
        reconstructed, _ = self.lstm(x)  # (B, T, output_dim)
        reconstructed = self.activation(reconstructed)
        return reconstructed


class HippocampalFormation(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, sparsity_weight: float = 1e-4):
        super().__init__()
        self.encoder = HippocampalEncoder(input_dim, latent_dim)
        self.decoder = HippocampalDecoder(latent_dim, input_dim)  # Output to input_dim (C)
        self.sparsity_weight = sparsity_weight

    def forward(self, x):
        B, T, C = x.shape
        # No need to broadcast last token; LSTM handles full sequence
        latent = self.encoder(x)  # (B, T, latent_dim)
        reconstructed = self.decoder(latent)  # (B, T, C)
        recon_loss = F.mse_loss(reconstructed, x)  # Reconstruct full input sequence
        sparsity_loss = self.sparsity_weight * torch.mean(torch.abs(latent))
        return reconstructed, recon_loss + sparsity_loss


class MiniGPT(nn.Module):

    def __init__(self, vocab_size: int, n_layer: int = n_layer, n_head: int = n_head, head_size: int = head_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd, device=device)
        self.position_embedding_table = nn.Embedding(sequence_length, n_embd, device=device)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, head_size) for _ in range(n_layer-1)])
        self.last_block = Block(n_embd, n_head, head_size)
        self.ln_f = nn.LayerNorm(n_embd, device=device)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # Add SAE(Hippocampal Formation)
        latent_dim = n_embd * 4  # Overcomplete dictionary
        self.hippocampal_formation = HippocampalFormation(n_embd, latent_dim)
        # Register hook on the second last block to capture activations
        self.activations = None
        def hook_fn(_, __, output):
            self.activations = output.detach()  # Capture residual output
        self.blocks[-1].register_forward_hook(hook_fn)  # Sampling after the second last block

    def forward(self, idx, targets=None):
        B, T = idx.shape
        emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = emb + pos_emb  # original input
        x = self.blocks(x)
        sae_loss = None
        if self.activations is not None:
            decoded, sae_loss = self.hippocampal_formation(self.activations)
            x = x + 0.1 * decoded[:, :x.shape[1], :]  # Inject as residual (truncate T if needed)
        x = self.last_block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss, sae_loss
    
    def generate(self, idx, max_new_tokens=500):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -sequence_length:]
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    
    def interactive_prompt(self, encode_fn, decode_fn, max_new_tokens=500, prompt=""):
        """
        Interactive console for prompting the model.
        :param encode_fn: Function to encode text to idx tensor (e.g., from dataloader)
        :param decode_fn: Function to decode idx tensor to text
        :param max_new_tokens: Tokens to generate per response
        :param prompt: Optional starting prompt
        """
        context = torch.tensor(encode_fn(prompt), dtype=torch.long, device=device).unsqueeze(0)  # (1, T)
        print("Interactive mode (type 'exit' to quit):")
        while True:
            user_input = input("")
            if user_input.lower() == 'exit':
                break
            # Append user input to context
            user_idx = torch.tensor(encode_fn(user_input), dtype=torch.long, device=device).unsqueeze(0)
            context = torch.cat([context, user_idx], dim=1)
            # Generate continuation
            generated = self.generate(context, max_new_tokens)
            response = decode_fn(generated[0][context.shape[1]:])  # Pass tensor slice directly (remove .tolist())
            print(f"Model: {response}")
            # Update context with generated tokens
            context = generated


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_interval):
    # for output
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_interval)
        sae_losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            if split == "train":
                X, Y = train_data.get_batch(batch_size=batch_size, block_size=sequence_length)
            else:
                X, Y = val_data.get_batch(batch_size=batch_size, block_size=sequence_length)
            _, loss, sae_loss = model(X, Y)
            losses[k] = loss.item()
            sae_losses[k] = sae_loss.item()
        out[split] = losses.mean()
        out[f"{split}_sae"] = sae_losses.mean()
    model.train()
    return out


def train_model(model, train_data, val_data, lr, max_iters, eval_interval, sae_weight=0.1):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for iter in range(max_iters):
        if iter % eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, eval_interval)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, train sae loss {losses['train_sae']:.4f}, val sae loss {losses['val_sae']:.4f}")
        X, Y = train_data.get_batch(batch_size=batch_size, block_size=sequence_length)
        logits, loss, sae_loss = model(X, Y)
        total_loss = loss + sae_weight * sae_loss if sae_loss is not None else loss
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward() 
        optimizer.step()

