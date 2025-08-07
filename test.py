import torch
from core.model import MiniGPT, train_model, device, sequence_length
from dataset.dataloader import create_dataset

lr = 1e-3
max_iters = 5000
eval_interval = 100
fine_tune_lr = 1e-3
fine_tune_max_iters = 500
fine_tune_eval_interval = 100


def train_phase(model_path):
    # model = MiniGPT(train.vocab_size)
    model = MiniGPT.from_pretrained(
        model_path,
        use_lora=False,
        vocab_size=train.vocab_size,
    )
    model.to(device)
    train_model(model, train, test, lr=lr, max_iters=max_iters, eval_interval=eval_interval)
    torch.save(model.state_dict(), model_path)


def test_phase(model_path):
    model = MiniGPT.from_pretrained(
        model_path,
        use_lora=False,
        vocab_size=train.vocab_size,
    )
    model.to(device)
    idx = model.generate(torch.zeros((1, sequence_length), dtype=torch.long, device=device), 500)
    print(train.decode(idx[0]))


def fine_tune_phase(model_path):
    # Load with LoRA enabled for fine-tuning
    model = MiniGPT.from_pretrained(
        model_path,
        use_lora=True,
        vocab_size=train.vocab_size,
        lora_rank=8,           # Optional: override default LoRA rank (lower = fewer params)
        lora_alpha=16,         # Optional: LoRA scaling factor
        lora_dropout=0.1       # Optional: dropout for LoRA
    )
    model.to(device)
    train_model(model, train, test, lr=fine_tune_lr, max_iters=fine_tune_max_iters, eval_interval=fine_tune_eval_interval)
    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    data_path = "dataset/enwik8"
    model_path = "model_saves/model.pth"
    train, test = create_dataset(data_path)
    train_phase(model_path)
    # test_phase(model_path)
    # fine_tune_phase(model_path)

