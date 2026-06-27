# Transformer From Scratch

A minimal Transformer encoder implemented with pure NumPy, used to classify SMS messages as spam or ham.

Built as a learning project , every layer is written by hand with explicit loops so we can see exactly what is happening at each step.

---

## What is implemented

| Component | Description |
|---|---|
| `matmul` | Matrix multiplication with explicit triple loop |
| `transpose` | 2-D matrix transpose |
| `softmax` | Row-wise numerically stable softmax |
| `layer_norm` | Per-token layer normalisation |
| `feedforward` | Two-layer MLP with ReLU |
| `attention` | Scaled dot-product attention |
| `multi_head_attention` | Split → attend per head → concatenate |
| `encoder_block` | MHA + residual + LayerNorm + FFN + residual + LayerNorm |
| `transformer_encoder` | Stack of N encoder blocks |
| `classify` | Mean-pool → linear → sigmoid |

---

## How to run

```bash
pip install -r requirements.txt
python transformer.py
```

The script downloads the SMS Spam dataset (~500 KB), runs a forward pass, and prints predictions for four sample messages.

> **Note:** weights are randomly initialised so no training is implemented. The demo validates the pipeline shape, not accuracy.

---

## Architecture

```
Text input
  │
  ▼  tokenise → embed
  │
  │  ┌──────────────────────────────┐
  │  │  Encoder Block × N           │
  │  │                              │
  │  │  X ──► Multi-Head Attention ─┐
  │  │        (self-attention)       │
  │  │  ┌─────── + X (residual) ◄──┘│
  │  │  │                            │
  │  │  ▼ LayerNorm                  │
  │  │                               │
  │  │  ──► FeedForward (ReLU) ──┐   │
  │  │  ┌──── + residual ◄───────┘   │
  │  │  │                            │
  │  │  ▼ LayerNorm                  │
  └──┴──────────────────────────────┘
  │
  ▼  mean-pool across sequence
  │
  ▼  linear + sigmoid → probability
```

---

## Requirements

- Python ≥ 3.8
- NumPy
- pandas (data loading only)

---

## Reference

Vaswani, A. et al. (2017). *Attention Is All You Need.* https://arxiv.org/abs/1706.03762
