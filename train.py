import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from datasets import load_dataset
from functools import partial

from src.data.dataset import TranslationDataset, collate_fn
from src.data.tokenizer import build_tokenizers
from src.model.transformer import Transformer

import wandb


def noam_lr(step, d_model, warmup_steps=4000):
    step = max(step, 1)
    return (d_model**-0.5)*min(step**-0.5, step*warmup_steps**-1.5)


def get_tokenizers(ds, vocab_size=8000, out_dir="tokenizers"):
    en_path = Path(out_dir) / "en.json"
    de_path = Path(out_dir) / "de.json"

    if en_path.exists() and de_path.exists():
        return Tokenizer.from_file(str(en_path)), Tokenizer.from_file(str(de_path))

    return build_tokenizers(ds, vocab_size, out_dir)


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = 512

    Path("checkpoints").mkdir(exist_ok=True)

    wandb.init(
        project="transformer-multi30k",
        config={
            "d_model": d_model,
            "num_heads": 8,
            "num_layers": 6,
            "d_ff": 2048,
            "batch_size": 64,
            "warmup_steps": 4000,
            "label_smoothing": 0.1,
        }
    )

    ds = load_dataset("bentrevett/multi30k")
    en_tok, de_tok = get_tokenizers(ds)
    pad_id=de_tok.token_to_id("<pad>")
    assert en_tok.token_to_id("<pad>")==pad_id
    train_ds = TranslationDataset(ds["train"], en_tok, de_tok)
    train_loader = DataLoader(train_ds, batch_size=64,
                              shuffle=True, collate_fn=partial(collate_fn,pad_id=pad_id))

    model = Transformer(src_vocab_size=en_tok.get_vocab_size(),
                        tgt_vocab_size=de_tok.get_vocab_size(),
                        d_model=d_model, pad_id=pad_id).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: noam_lr(step, d_model))
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=pad_id, label_smoothing=0.1)

    model.train()
    step = 0
    for epoch in range(50):
        total_loss = 0.0
        for batch in train_loader:
            src = batch["src"].to(device)  # (B, S_src)
            tgt_input = batch["tgt_input"].to(device)  # (B, S_tgt)
            tgt_output = batch["tgt_output"].to(device)

            logits = model(src, tgt_input)  # (B, S_tgt, tgt_vocab_size)
            loss = criterion(logits.reshape(-1, logits.size(-1)),
                             tgt_output.reshape(-1))  # (B*S, vocab), (B*S, )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            wandb.log({"train/loss": loss.item(),
                      "train/lr": scheduler.get_last_lr()[0], }, step=step)
            step += 1

            total_loss += loss.item()
        avg_loss = total_loss/len(train_loader)
        print(f"epoch {epoch} | avg loss {avg_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_loss}, step=step)
        torch.save(model.state_dict(),
                   f"checkpoints/checkpoint_epoch{epoch}.pt")

        artifact = wandb.Artifact("transformer-checkpoint", type="model")
        artifact.add_file(f"checkpoints/checkpoint_epoch{epoch}.pt")
        wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    train()
