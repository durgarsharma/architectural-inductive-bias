# Attention Cannot Track Cycles: Mechanistic Evidence from Vigen\`{e}re Cipher Decryption
---

## Overview

This paper presents a mechanistic investigation of a specific and reproducible Transformer failure, the inability to track periodic positional structure. Using Vigenère cipher decryption as a controlled diagnostic, where success requires computing `position mod key_length` at every step, we assemble a chain of five experiments that rule out alternative explanations and converge on a precise mechanistic account.

**The core finding:** Transformers fail on Vigenère (9.88% word accuracy) while BiLSTMs succeed near perfectly (99.91%), not because of poor hyperparameters, wrong positional encodings, or an unusual key length, but because the attention mechanism cannot form the modular positional structure that cycle tracking requires. Models trained under this constraint fall back on character frequency analysis, a qualitatively wrong strategy that degrades predictably as key length grows.

---

## Key Results

### 1. The Failure Is Large and Robust to Configuration
- BiLSTM achieves **99.91%** word accuracy on Vigenère; Transformer achieves **9.88%**
- **24 hyperparameter configurations** (layers 2–8, d_model 128–512, sinusoidal and learned PE) all yield ~39% character accuracy, a 60pp gap from BiLSTM
- Transformer and MLP track each other within 0.2pp at every configuration, showing global attention offers no advantage over a position agnostic baseline

### 2. Oracle Positional Encoding Does Not Fix the Failure
- Five PE variants tested: sinusoidal, NoPE, learned, RoPE, and a custom **oracle modular encoding** that directly provides `sin/cos(2π·(t mod K)/K)` as input features
- Oracle PE reaches **59.57%**, still 40pp below BiLSTM
- Causal (left-to-right) attention alone: **+0.67pp**, no effect
- Causal + RoPE: **57.24%**, best Transformer condition, still 42.66pp below BiLSTM
- The bottleneck is the attention mechanism itself, not the positional signal available to it

### 3. The Failure Scales With Cycle Complexity

| Key Length K | BiLSTM | Transformer | CNN | MLP | Gap (pp) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 99.95 | 73.30 | 99.08 | 73.10 | 26.7 |
| 3 | 99.91 | 63.40 | 98.11 | 63.35 | 36.5 |
| 6 | 99.89 | 39.22 | 94.04 | 39.14 | 60.7 |
| 9 | 99.82 | 42.27 | 89.58 | 42.11 | 57.5 |
| 12 | 99.75 | 34.06 | 85.26 | 33.86 | 65.7 |
| 18 | 99.75 | 29.25 | 77.26 | 29.27 | 70.5 |

### 4. Linear Probing Identifies the Mechanism

| K | Chance | BiLSTM task | BiLSTM probe | TF task | TF probe | Probe gap |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 2 | 50.0% | 99.95% | 99.89% | 73.28% | 72.32% | 27.6pp |
| 3 | 33.3% | 99.91% | 99.88% | 63.39% | 62.32% | 37.6pp |
| 6 | 16.7% | 99.84% | 99.82% | 39.20% | 37.28% | 62.5pp |
| 9 | 11.1% | 99.79% | 99.82% | 42.27% | 26.28% | 73.5pp |
| 12 | 8.3% | 99.64% | 99.53% | 34.04% | 20.96% | 78.6pp |
| 18 | 5.6% | 99.57% | 99.39% | 29.29% | 13.96% | 85.4pp |

**The diagnostic finding:** At K≥9, Transformer probe accuracy falls *below* task accuracy, the model cannot be using cycle phase as its primary feature. It is doing character frequency analysis, not cycle tracking.

---

## Experiments

| S.No. | Experiment | Key Finding |
|:---|:---|:---|
| 1 | Attention visualisation + linear probe at K=6 | BiLSTM probe: 99.9%; Transformer probe: 37.1% |
| 2 | Hyperparameter grid search (24 configs) | All configs ~39.4%; scaling does not help |
| 3 | Positional encoding ablations (5 variants) | Oracle PE reaches 59.6%; bottleneck is attention |
| 4 | Key length variation (K=2–18) | Gap grows 26.7pp → 70.5pp monotonically |
| 5 | Linear probing across all key lengths | Probe falls below task accuracy at K≥9 |
| 6 | Causal masking ablation | +0.67pp alone; causal+RoPE +18pp; gap remains |

---

## Dataset

The Vigenère diagnostic suite uses **1,000 English Wikipedia articles** encrypted with configurable keywords with keyword CIPHER (K = 6), using a 70/15/15 train/validation/test split (700/150/150 articles). All text is lowercased and tokenised at the character level using 38 tokens: 26 lowercase letters, 10 digits, and two special tokens <PAD> and <UNK>. Sequences are truncated or padded to 512 characters.

For key length experiments, the same articles are encrypted again using:

| K | Keyword |
|:---:|:---:|
| 2 | AB |
| 3 | CAT |
| 6 | CIPHER |
| 9 | SECRETKEY |
| 12 | CRYPTOGRAPHY |
| 18 | SECRETCRYPTOGRAPHY |

---

## Models

| Model | Architecture | Role |
|:---|:---|:---|
| BiLSTM | 2-layer bidirectional, 256 hidden units/direction | Primary reference: solves Vigenère via sequential counting |
| Transformer | 4 layers, 8 heads, d_model=256, sinusoidal PE | Primary subject: fails to form modular positional structure |
| Char-CNN | Parallel convolutions, kernel sizes {3,5,7}, 256 filters | Local pattern baseline: degrades as K exceeds receptive field |
| MLP | 3 FC layers (512, 256, 128), character-independent | Frequency analysis lower bound |

---

## Ciphers

| Cipher | Type | Task requirement |
|:---|:---|:---|
| Caesar | Monoalphabetic | Fixed shift | no positional reasoning needed |
| Atbash | Monoalphabetic | Alphabet reversal | no positional reasoning needed |
| Affine | Monoalphabetic | Linear map | no positional reasoning needed |
| Vigenère | Polyalphabetic | Track `t mod K` | requires cycle phase encoding |
| Substitution (Fixed) | Monoalphabetic | Arbitrary fixed map | no positional reasoning needed |
| Substitution (Random) | Random per article | Negative control | no learnable structure |
| AES-256 / DES | Modern block cipher | Negative control | cryptographically secure |

Monoalphabetic ciphers serve as **positive controls**, all architectures solve them, confirming the Vigenère divergence is specific to periodicity.
