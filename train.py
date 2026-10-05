import math
from pathlib import Path

import torch
import argparse
import random
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from datasets import load_dataset
from functools import partial

from src.data.dataset import TranslationDataset, collate_fn
from src.data.tokenizer import build_tokenizers
from src.model.transformer import Transformer
from src.model.torch_baseline import TorchTransformer

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

def run_epoch_val(model, loader, criterion, pad_id, device):
    model.eval()
    total_loss, total_tokens=0.0, 0
    with torch.no_grad():
        for batch in loader:
            src=batch["src"].to(device)
            tgt_input=batch["tgt_input"].to(device)
            tgt_output=batch["tgt_output"].to(device)
            logits=model(src, tgt_input)
            loss=criterion(logits.reshape(-1,logits.size(-1)),tgt_output.reshape(-1))
            n=(tgt_output!=pad_id).sum().item()
            total_loss+=loss.item()*n
            total_tokens+=n
    return total_loss /total_tokens


def train(seed, post_ln, baseline):
    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = 512

    variant="torch" if baseline else ("post" if post_ln else "pre")
    Path("checkpoints").mkdir(exist_ok=True)
    ckpt_path=f"checkpoints/{variant}_seed{seed}.pt"

    wandb.init(
        project="transformer-multi30k",
        group=f"{variant}-ln",
        name=f"{variant}-ln-seed{seed}",
        config={
            "d_model": d_model,
            "num_heads": 8,
            "num_layers": 6,
            "d_ff": 2048,
            "batch_size": 64,
            "warmup_steps": 4000,
            "label_smoothing": 0.1,
            "seed":seed,
            "norm_first": not post_ln
        }
    )

    ds = load_dataset("bentrevett/multi30k")
    en_tok, de_tok = get_tokenizers(ds)
    pad_id=de_tok.token_to_id("<pad>")
    assert en_tok.token_to_id("<pad>")==pad_id
    collate=partial(collate_fn, pad_id=pad_id)
    train_loader = DataLoader(TranslationDataset(ds["train"], en_tok, de_tok), batch_size=64,
                              shuffle=True, collate_fn=collate, generator=torch.Generator().manual_seed(seed))
    val_loader=DataLoader(TranslationDataset(ds["validation"],en_tok,de_tok),batch_size=64,collate_fn=collate)

    if baseline:
        model = TorchTransformer(src_vocab_size=en_tok.get_vocab_size(),
                                 tgt_vocab_size=de_tok.get_vocab_size(),
                                 d_model=d_model, pad_id=pad_id).to(device)
    else:
        model = Transformer(src_vocab_size=en_tok.get_vocab_size(),
                            tgt_vocab_size=de_tok.get_vocab_size(),
                            d_model=d_model, pad_id=pad_id, norm_first=not post_ln).to(device)
    wandb.summary["params"] = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: noam_lr(step, d_model))
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=pad_id, label_smoothing=0.1)

    step = 0
    best_val=float("inf")
    for epoch in range(50):
        model.train()
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
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            wandb.log({"train/loss": loss.item(),
                      "train/lr": scheduler.get_last_lr()[0],
                      "train/grad_norm":grad_norm.item()}, step=step)
            step += 1

            total_loss += loss.item()
        avg_loss = total_loss/len(train_loader)
        val_loss=run_epoch_val(model, val_loader, criterion, pad_id, device)
        print(f"epoch {epoch} | train loss {avg_loss:.4f} | val {val_loss:.4f}")
        wandb.log({"train/epoch_loss": avg_loss, "val/loss":val_loss},step=step)
        
        if val_loss < best_val:
            best_val=val_loss
            torch.save(model.state_dict(),ckpt_path)

    artifact = wandb.Artifact(f"transformer-{variant}-seed{seed}", type="model")
    artifact.add_file(ckpt_path)
    wandb.log_artifact(artifact)
    wandb.summary["best_val_loss"]=best_val

    wandb.finish()


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--post-ln",action="store_true")
    parser.add_argument("--baseline",action="store_true")
    args=parser.parse_args()
    train(args.seed,args.post_ln,args.baseline)
