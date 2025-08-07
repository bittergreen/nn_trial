import torch
from torch.utils.data import Dataset


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class TextDataset(Dataset):

    def __init__(self, text, chars, max_length):
        self.text = text
        self.chars = chars
        self.vocab_size = len(self.chars)
        self.max_length = max_length  # the length of the whole text
        self.stoi = {s: i for i, s in enumerate(chars)}
        self.itos = {i: s for s, i in self.stoi.items()}

    def encode(self, x):
        return torch.tensor([self.stoi[s] for s in x])

    def decode(self, x):
        return "".join([self.itos[i.item()] for i in x])

    def __getitem__(self, idx):
        return self.encode(self.text[idx])

    def get_batch(self, batch_size, block_size):
        ix = torch.randint(self.max_length - block_size, (batch_size, ))
        x = torch.stack([self[i: i+block_size] for i in ix])
        y = torch.stack([self[i+1: i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y


def create_dataset(input_file):
    with open(input_file) as f:
        raw = f.read()
    chars = sorted(list(set(raw)))
    length = len(raw)
    print(f"List of chars: {''.join(chars)}")
    print(f"Num of chars: {len(chars)}")
    print(f"Length of text: {len(raw)}")
    n = int(0.9*length)
    train_set = TextDataset(raw[:n], chars, n)
    test_set = TextDataset(raw[n:], chars, length-n)
    return train_set, test_set


