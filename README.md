# Transformer from Scratch — Multi30k En→De

An encoder–decoder Transformer implemented from scratch in PyTorch and trained on Multi30k English→German translation.
**`torch.nn.Transformer` / `nn.MultiheadAttention` are not used in the model** — attention, multi-head projection, masking, positional encoding, residual/LayerNorm blocks, and the encoder/decoder stacks are all hand-written. `nn.Transformer` appears only in a separate baseline used to check the implementation.

On top of the implementation, the project runs a small controlled experiment: **Pre-LN vs Post-LN**, 3 seeds each, compared on convergence, gradient norm, validation loss, and BLEU.

> **Note.** The default model is a **Pre-LN variant**. The original *Attention Is All You Need* uses Post-LN; Post-LN is available via `--post-ln`. This is not an exact reproduction of the paper (see [Limitations](#limitations)).

## Results

Multi30k test set (1,000 sentences), greedy decoding, sacreBLEU. Mean ± std over seeds 0, 1, 2. Each run uses the checkpoint with the lowest validation loss.

| Model | Params | Best val loss | Test BLEU (seeds 0/1/2) | **Test BLEU** |
|---|---|---|---|---|
| Ours, Pre-LN | 56.44M | 3.913 ± 0.004 | 18.98 / 16.88 / 19.63 | **18.50 ± 1.44** |
| Ours, Post-LN | 56.43M | 3.579 ± 0.009 | 24.22 / 26.05 / 24.08 | **24.78 ± 1.10** |
| `nn.Transformer` baseline, Pre-LN | 56.44M | 3.880 ± 0.003 | 18.18 / 18.44 / 11.63 | **16.08 ± 3.86** |

Validation loss includes label smoothing (ε = 0.1).

### 1. Does the implementation behave correctly? (Ours Pre-LN vs `nn.Transformer`)

The baseline (`src/model/torch_baseline.py`) shares the embeddings, positional encoding, and output projection with our model and swaps only the encoder/decoder stacks for `nn.Transformer(norm_first=True)`. Same data, tokenizer, hyperparameters, and seeds.

- **Identical parameter count** (56,436,544), so the architectures match.
- **Validation curves coincide**: best val loss 3.913 vs 3.880, and the best epoch is the same per seed (9 / 7 / 9).
- **Gradient norms coincide**: mean pre-clip norm 0.88 vs 0.88 during warmup, 0.72 vs 0.72 afterwards.
- **BLEU difference is not significant**: Welch t = 1.02, df ≈ 2.5, p ≈ 0.40. Baseline seed 2 (11.63) degenerated into repetition loops under greedy decoding; without it the baseline averages 18.31.

The remaining gap is attributable to initialization: `nn.Transformer` re-initializes with Xavier-uniform, while our layers use PyTorch's default `nn.Linear` init.

### 2. Pre-LN vs Post-LN

| | Pre-LN | Post-LN |
|---|---|---|
| Mean grad norm, first 4k steps (warmup) | 0.88 | 3.08 |
| Mean grad norm, after warmup | 0.72 | 2.19 |
| Max grad norm | 3.1 – 3.5 | 39 – 347 |
| Steps clipped (norm > 1.0) | 13% | 87% |
| Best epoch | 7 – 9 | 10 – 11 |
| Best val loss | 3.913 | 3.579 |
| Test BLEU | 18.50 ± 1.44 | 24.78 ± 1.10 |

- **Post-LN is +6.3 BLEU better** (Welch t = 6.02, df ≈ 3.7, p ≈ 0.005), with a matching gap in validation loss.
- **Pre-LN has much smaller, more stable gradients**, while Post-LN shows large spikes (up to 347) and is clipped on almost every step. This matches the usual picture: Pre-LN is easier to optimize, while Post-LN, when it trains stably, often reaches better final quality.
- **Caveats.** (1) The hyperparameters (Noam schedule, 4k warmup) come from the original paper, which was tuned for Post-LN. Pre-LN usually tolerates and benefits from a larger learning rate or shorter warmup, which was not tuned here. (2) Because 87% of Post-LN steps are clipped, gradient clipping substantially shapes its effective update. The conclusion is "Post-LN wins **in this setup**", not in general.

### Training curves

W&B project: <https://wandb.ai/models-seoul-national-university3062/transformer-multi30k> (runs grouped by `pre-ln` / `post-ln` / `torch-ln`).

<!-- TODO: export val/loss and train/grad_norm panels from W&B (panel ⋯ → Export PNG) into docs/ and embed:
![val loss](docs/val_loss.png)
![grad norm](docs/grad_norm.png)
-->

All runs overfit early: validation loss bottoms out around epochs 7–11 and rises afterwards (Pre-LN 3.91 → 4.16 by epoch 50) while training loss keeps falling to ~1.38. Checkpoint selection by validation loss is what protects the reported numbers.

### Example translations

Post-LN, seed 1. These are the **first three test sentences**, not cherry-picked:

| Source | Reference | Ours |
|---|---|---|
| A man in an orange hat starring at something. | Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt. | Ein Mann mit einer orangefarbenen Kopfbedeckung starrt auf etwas. |
| A Boston Terrier is running on lush green grass in front of a white fence. | Ein Boston Terrier läuft über saftig-grünes Gras vor einem weißen Zaun. | Ein weißer Rennautos an einem grünen Zaun auf einer grünen Wiese. |
| A girl in karate uniform breaking a stick with a front kick. | Ein Mädchen in einem Karateanzug bricht ein Brett mit einem Tritt. | Ein Mädchen mit einem Karateanzug macht vor einem Karate-Klache. |

Full outputs (first 5 sentences per run) are in `logs/eval_*.log`.

## What is implemented

| Module | File | Notes |
|---|---|---|
| Scaled dot-product attention | `src/model/attention.py` | `softmax(QKᵀ/√d_k)V`, boolean mask (True = blocked) |
| Multi-head attention | `src/model/attention.py` | Separate Q/K/V/output projections, head split/merge |
| Position-wise feed-forward | `src/model/layers.py` | Linear → ReLU → Dropout → Linear |
| Residual + LayerNorm | `src/model/layers.py` | `SublayerConnection`, Pre-LN or Post-LN via `norm_first` |
| Sinusoidal positional encoding | `src/model/positional.py` | |
| Encoder / Decoder stacks | `src/model/encoder.py`, `decoder.py` | Final LayerNorm only in Pre-LN |
| Full model + masks | `src/model/transformer.py` | Source padding mask; target padding + causal mask; embeddings scaled by √d_model |
| BPE tokenizers | `src/data/tokenizer.py` | One per language, HF `tokenizers`, Metaspace pre-tokenizer |
| Training loop | `train.py` | Noam LR schedule, label smoothing, grad clipping, validation, best checkpoint, W&B logging |
| Greedy decoding + BLEU | `evaluate.py` | sacreBLEU corpus BLEU |

## Setup

| | |
|---|---|
| Dataset | [Multi30k](https://huggingface.co/datasets/bentrevett/multi30k) En→De: 29,000 train / 1,014 val / 1,000 test |
| Tokenizer | BPE, vocab 8,000 per language, Metaspace pre-tokenizer (lossless round-trip) |
| Model | 6 + 6 layers, d_model 512, 8 heads, d_ff 2048, dropout 0.1 |
| Optimizer | Adam (β = 0.9, 0.98, ε = 1e-9), Noam schedule, 4,000 warmup steps |
| Training | Batch 64, 50 epochs (~22.7k steps), label smoothing 0.1, grad-norm clip 1.0 |
| Checkpoint | Lowest validation loss |
| Decoding | Greedy, max 128 tokens |
| Hardware | <!-- TODO: GPU model --> single GPU (Vast.ai), ~30–50 min per run |
| Software | Python 3.12, PyTorch, `uv` |

## Reproduce

```bash
uv sync
uv run python -m src.data.tokenizer            # trains tokenizers/en.json, de.json

# train: --post-ln for Post-LN, --baseline for nn.Transformer
uv run python train.py --seed 0
uv run python train.py --seed 0 --post-ln
uv run python train.py --seed 0 --baseline

# evaluate (pass the same flag used for training)
uv run python evaluate.py --ckpt checkpoints/pre_seed0.pt
uv run python evaluate.py --ckpt checkpoints/post_seed0.pt --post-ln
uv run python evaluate.py --ckpt checkpoints/torch_seed0.pt --baseline
```

Checkpoints are saved to `checkpoints/{pre,post,torch}_seed{N}.pt` and uploaded to W&B as artifacts.

## Code structure

```
.
├── train.py                  # training loop (--seed, --post-ln, --baseline)
├── evaluate.py               # greedy decoding + test BLEU
├── src/
│   ├── data/
│   │   ├── tokenizer.py      # BPE tokenizer training
│   │   └── dataset.py        # Dataset + padding collate
│   └── model/
│       ├── attention.py      # scaled dot-product + multi-head attention
│       ├── layers.py         # feed-forward, residual/LayerNorm (Pre/Post-LN)
│       ├── positional.py     # sinusoidal positional encoding
│       ├── encoder.py
│       ├── decoder.py
│       ├── transformer.py    # full model + masks
│       └── torch_baseline.py # nn.Transformer baseline (for verification only)
└── logs/                     # test BLEU + sample translations per run
```

## Limitations

- **Not an exact reproduction** of *Attention Is All You Need*: the default is Pre-LN; no weight tying between embeddings and output projection; greedy decoding instead of beam search; no checkpoint averaging; trained on Multi30k, not WMT14.
- **Absolute BLEU is low** compared with typical Multi30k results. Contributing factors visible in the logs: strong overfitting (56M params on 29k sentence pairs; best epoch ≈ 10 of 50), greedy decoding that sometimes falls into repetition loops, and no hyperparameter tuning.
- **Hyperparameters were not tuned per variant**, so the Pre-LN vs Post-LN result holds for the paper's schedule only.
- **3 seeds** give a rough estimate of variance; the t-tests above have very few degrees of freedom.
- **No unit tests**; correctness is argued from agreement with `nn.Transformer` (parameter count, loss and gradient-norm curves, BLEU).
- Training is not bit-for-bit deterministic on GPU (fixed seeds, but no `torch.use_deterministic_algorithms`).
