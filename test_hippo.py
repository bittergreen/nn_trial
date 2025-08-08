from core.model4 import MiniGPT, train_model, device, sequence_length
from dataset.dataloader import create_dataset
import torch

lr = 1e-5  # 1e-4 for the first 10000 steps, then 1e-5
max_iters = 2000
eval_interval = 100


def train_phase(model_path):
    model = MiniGPT(train.vocab_size)
    model.load_state_dict(torch.load(model_path))  # for continue training
    model.to(device)
    train_model(model, train, test, lr=lr, max_iters=max_iters, eval_interval=eval_interval)
    torch.save(model.state_dict(), model_path)


def test_phase(model_path):
    model = MiniGPT(train.vocab_size)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    idx = model.generate(torch.zeros((1, sequence_length), dtype=torch.long, device=device), 500)
    print(train.decode(idx[0]))


if __name__ == "__main__":
    data_path = "dataset/tiny-shakespeare.txt"
    model_path = "model_saves/hippo_model.pth"
    train, test = create_dataset(data_path)
    # train_phase(model_path)
    test_phase(model_path)
