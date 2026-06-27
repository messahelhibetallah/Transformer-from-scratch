import re
from collections import Counter
from typing import List

import numpy as np
import pandas as pd



def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication C = A @ B implemented with explicit loops.

    Args:
        A: (m, k)
        B: (k, n)

    Returns:
        C: (m, n)

    Raises:
        ValueError: if inner dimensions do not match.
    """
    if A.shape[1] != B.shape[0]:
        raise ValueError(
            f"Shape mismatch for matmul: {A.shape} cannot multiply {B.shape}"
        )
    C = np.zeros((A.shape[0], B.shape[1]))
    for a in range(A.shape[0]):
        for b in range(B.shape[1]):
            for k in range(B.shape[0]):
                C[a][b] += A[a][k] * B[k][b]
    return C


def transpose(A: np.ndarray) -> np.ndarray:
    """
    Transpose a 2-D matrix.

    Args:
        A: (m, n)

    Returns:
        (n, m)
    """
    B = np.zeros((A.shape[1], A.shape[0]))
    for a in range(A.shape[0]):
        for b in range(A.shape[1]):
            B[b][a] = A[a][b]
    return B


def softmax(x: np.ndarray) -> np.ndarray:
    """
    Numerically stable softmax applied **row-wise** over the last dimension.

    Each row of the attention score matrix must sum to 1 independently.
    Applying softmax over the whole flattened matrix (the original bug)
    makes every row sum to a fraction instead of 1, breaking attention.

    Args:
        x: array of any shape (..., n)

    Returns:
        softmax probabilities, same shape as x
    """
    # Subtract row-max for numerical stability before exponentiating
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Layer normalisation applied **per row** (i.e. per token position).

    The original implementation took the mean and variance of the *entire*
    matrix, which mixed information across tokens and produced incorrect
    normalisation when the sequence had more than one token.

    Args:
        x:   (seq_len, d_model)
        eps: small constant to avoid division by zero

    Returns:
        normalised array of the same shape
    """
    mean = np.mean(x, axis=-1, keepdims=True)   # (seq_len, 1)
    var  = np.var(x,  axis=-1, keepdims=True)   # (seq_len, 1)
    return (x - mean) / np.sqrt(var + eps)


def feedforward(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """
    Position-wise feed-forward network: Linear → ReLU → Linear.

    Args:
        x:  (seq_len, d_model)
        W1: (d_model, d_ff)
        b1: (d_ff,)
        W2: (d_ff, d_model)
        b2: (d_model,)

    Returns:
        (seq_len, d_model)
    """
    hidden = np.maximum(0, matmul(x, W1) + b1)   # ReLU
    return matmul(hidden, W2) + b2

 
# Attention

def attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
) -> np.ndarray:
    """
    Scaled dot-product attention.

    score = softmax( Q @ K^T / sqrt(d_k) ) @ V

    Args:
        Q: (seq_len, d_k)
        K: (seq_len, d_k)
        V: (seq_len, d_k)

    Returns:
        (seq_len, d_k)
    """
    d_k = K.shape[1]
    score = matmul(Q, transpose(K)) / np.sqrt(d_k)   # (seq_len, seq_len)
    weights = softmax(score)                           # row-wise softmax
    return matmul(weights, V)


def multi_head_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """
    Multi-head attention: split Q/K/V along the embedding dimension, run
    one attention head per slice, then concatenate.

    Args:
        Q:         (seq_len, d_model)
        K:         (seq_len, d_model)
        V:         (seq_len, d_model)
        num_heads: number of heads (d_model must be divisible by num_heads)

    Returns:
        (seq_len, d_model)
    """
    d_model = K.shape[1]
    d_k = d_model // num_heads
    heads = []
    for i in range(num_heads):
        Q_i = Q[:, i * d_k : (i + 1) * d_k]
        K_i = K[:, i * d_k : (i + 1) * d_k]
        V_i = V[:, i * d_k : (i + 1) * d_k]
        heads.append(attention(Q_i, K_i, V_i))
    return np.concatenate(heads, axis=-1)   # (seq_len, d_model)


# Encoder

def encoder_block(
    X: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """
    Single Transformer encoder block.

    Architecture (Pre-norm is used in modern practice, but this follows the
    original paper's Post-norm order to match the notebook's intent):

        X → MHA → residual → LayerNorm → FFN → residual → LayerNorm

    Args:
        X:         (seq_len, d_model) input
        W1, b1:    first FFN layer weights / biases
        W2, b2:    second FFN layer weights / biases
        num_heads: attention heads

    Returns:
        (seq_len, d_model)
    """
    # Self-attention sub-layer
    mha_out   = multi_head_attention(X, X, X, num_heads)
    residual1 = X + mha_out
    norm1     = layer_norm(residual1)

    # Feed-forward sub-layer
    ffn_out   = feedforward(norm1, W1, b1, W2, b2)
    residual2 = norm1 + ffn_out
    norm2     = layer_norm(residual2)

    return norm2


def transformer_encoder(
    X: np.ndarray,
    W1_list: list,
    b1_list: list,
    W2_list: list,
    b2_list: list,
    num_heads: int,
) -> np.ndarray:
    """
    Stack of encoder blocks applied sequentially.

    Each block receives the *output* of the previous block — not the original
    input X. The original code passed X every iteration, so the blocks did
    not chain and only the last block's transformation was used.

    Args:
        X:          (seq_len, d_model) input embeddings
        W1_list:    list of (d_model, d_ff) weight matrices, one per block
        b1_list:    list of (d_ff,) biases
        W2_list:    list of (d_ff, d_model) weight matrices
        b2_list:    list of (d_model,) biases
        num_heads:  number of attention heads

    Returns:
        (seq_len, d_model) contextualised representations
    """
    out = X
    for i in range(len(W1_list)):
        out = encoder_block(out, W1_list[i], b1_list[i], W2_list[i], b2_list[i], num_heads)
    return out


# Classification head 

def classify(
    encoder_output: np.ndarray,
    W_out: np.ndarray,
    b_out: float,
) -> float:
    """
    Binary classification head.

    Mean-pool the encoder output across the sequence dimension, then apply
    a linear layer followed by sigmoid to get a probability in (0, 1).

    Args:
        encoder_output: (seq_len, d_model)
        W_out:          (d_model,) weight vector
        b_out:          scalar bias

    Returns:
        probability (float) — > 0.5 → Spam, ≤ 0.5 → Ham
    """
    pooled = np.mean(encoder_output, axis=0)        # (d_model,)
    logit  = np.dot(pooled, W_out) + b_out
    return 1.0 / (1.0 + np.exp(-logit))



def load_data() -> pd.DataFrame:
    """Download the SMS Spam dataset."""
    url = (
        "https://raw.githubusercontent.com/justmarkham/"
        "pycon-2016-tutorial/master/data/sms.tsv"
    )
    data = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])
    print(f"Loaded {len(data)} messages  ({data['label'].value_counts().to_dict()})")
    return data


def build_vocab(messages: List[List[str]], vocab_size: int = 2000) -> dict:
    """Build a word-to-index vocabulary from tokenised messages."""
    word_counts: Counter = Counter()
    for msg in messages:
        for word in msg:
            word_counts[word] += 1

    vocab: dict = {}
    for word, _ in word_counts.most_common(vocab_size):
        vocab[word] = len(vocab)

    vocab.setdefault("<UNK>", len(vocab))
    vocab.setdefault("<PAD>", len(vocab))
    return vocab


def tokenise(text: str) -> List[str]:
    """Lowercase and split on non-word characters."""
    return re.sub(r"\W+", " ", text.lower()).split()


def message_to_indices(
    tokens: List[str],
    vocab: dict,
    seq_len: int = 20,
) -> List[int]:
    """
    Convert a token list to a fixed-length index sequence.

    Truncates sequences that are too long; pads short ones with <PAD>.
    """
    pad_id = vocab["<PAD>"]
    unk_id = vocab["<UNK>"]
    indices = [vocab.get(tok, unk_id) for tok in tokens]
    indices = indices[:seq_len]
    indices += [pad_id] * (seq_len - len(indices))
    return indices


def indices_to_embedding(
    indices: List[int],
    embeddings: np.ndarray,
) -> np.ndarray:
    """Look up embedding vectors for a list of token indices."""
    return np.array([embeddings[i] for i in indices])

# Demo

def main() -> None:
    print("=" * 60)
    print("Transformer Encoder — spam classifier demo")
    print("=" * 60)

    # Hyper-parameters
    d_model    = 8
    d_ff       = 16
    num_heads  = 2
    num_blocks = 2
    seq_len    = 20
    vocab_size = 2000

    data     = load_data()
    messages = [tokenise(msg) for msg in data["message"]]
    vocab    = build_vocab(messages, vocab_size)
    actual_vocab_size = len(vocab)

    # ── Random (untrained) parameters ────────────────────────────────────────
    rng       = np.random.default_rng(42)
    embedding_matrix = rng.random((actual_vocab_size, d_model))

    W1_list = [rng.random((d_model, d_ff))  for _ in range(num_blocks)]
    b1_list = [rng.random(d_ff)             for _ in range(num_blocks)]
    W2_list = [rng.random((d_ff, d_model))  for _ in range(num_blocks)]
    b2_list = [rng.random(d_model)          for _ in range(num_blocks)]

    W_out = rng.random(d_model)
    b_out = 0.0

    def predict(message: str) -> tuple:
        tokens  = tokenise(message)
        indices = message_to_indices(tokens, vocab, seq_len)
        X       = indices_to_embedding(indices, embedding_matrix)
        encoded = transformer_encoder(X, W1_list, b1_list, W2_list, b2_list, num_heads)
        prob    = classify(encoded, W_out, b_out)
        label   = "Spam" if prob > 0.5 else "Ham"
        return label, float(prob)

    # ── Run on a few samples ──────────────────────────────────────────────────
    samples = [
        "Don't forget the meeting tomorrow",
        "WINNER!! Claim your free prize now! Call 08001234",
        "Hey, are you free for lunch?",
        "Urgent! Your account has been compromised. Click here.",
    ]

    print()
    for msg in samples:
        label, score = predict(msg)
        print(f"  [{label:4s}  {score:.3f}]  {msg}")

    print()
    print("Note: weights are random — no training has been done.")
    print("This demo validates the pipeline shape, not accuracy.")


if __name__ == "__main__":
    main()