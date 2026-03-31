# FASEROH GSoC 2026 — Taylor Expansion via Seq2Seq

Evaluation test for the ML4SCI FASEROH project, GSoC 2026.

**Candidate:** Hivansh Dhakne — hivanshd@gmail.com

---

## What this is

The task is: given a symbolic function `f(x)`, predict its 4th-order Maclaurin (Taylor) expansion. Framed as a seq2seq problem where both input and output are sequences of math tokens.

```
sin(x)        ->  x - x**3/6
x * exp(x)    ->  x**4/6 + x**3/2 + x**2 + x
sin(x)*cos(x) ->  -x**3/3 + x
```

---

## Notebook walkthrough

The main file is `FASEROH_final_annotated.ipynb`. Run top to bottom on Colab (GPU recommended, ~15 min).

### Dataset generation

6,000 pairs generated with SymPy's `series().removeO()`. Three categories of expressions: base elementary functions, linear combinations, and products. Products are the hardest case since the Taylor expansion of `f*g` is the Cauchy product of their coefficient sequences, not just multiplying the individual expansions term-by-term. The dataset is deliberately small (4,200 train) to make the architectural comparison meaningful rather than just data-limited.

### Tokenisation

Regex tokeniser that treats math symbols as atomic tokens -- `sin`, `cos`, `**`, digits, operators. Shared vocabulary of 50 tokens for both encoder and decoder. One unexpected artifact: large integers like `128` and `256` appear in the vocab because SymPy's intermediate computations for complex products occasionally produce them as coefficients. They're handled correctly but it's a minor messiness.

### LSTM Seq2Seq

Bidirectional LSTM encoder (2 layers, hidden=256) with Bahdanau additive attention. Teacher-forcing ratio decays linearly from 0.50 to 0.05 over 30 epochs. The decay matters -- keeping teacher forcing high early stabilizes training, decaying it forces the model to handle its own predictions at test time.

Training was clean. Val loss dropped steadily from 2.27 to 0.11 over 30 epochs with no signs of overfitting.

### Transformer Seq2Seq

Standard `nn.Transformer` with sinusoidal positional encoding, 3 encoder/decoder layers, 4 heads, d_model=128. Label smoothing 0.10, cosine annealing LR.

The val loss plateaued at ~0.735 after epoch 5 and barely moved for the remaining 25 epochs. Not convergence -- the LR schedule was likely too aggressive early in training before attention weights could stabilize.

### Results

| Model | Exact Match | Token Acc | BLEU-4 | R^2 |
|---|---|---|---|---|
| LSTM (greedy) | 0.9156 | 0.9630 | 0.9718 | 0.8939 |
| Transformer v1 (greedy) | 0.9133 | 0.9366 | 0.9664 | 0.8754 |

LSTM edges out the Transformer on every metric. At this data scale (4,200 training examples), the recurrent inductive bias is genuinely useful -- the sequential structure of math expressions maps naturally onto recurrence. The Transformer matches it with 3x fewer parameters though, which suggests the architecture is appropriate and would close the gap with more data.

The R^2 being lower than exact match (0.894 vs 0.916 for LSTM) is worth noting. Most errors are coefficient mistakes -- predicting `x - x**3/7` instead of `x - x**3/6` -- which count as 100% wrong under exact match but are numerically close on [-0.5, 0.5].

### Improvement attempts

**Beam search (k=5)** on the trained Transformer: exact match dropped to 13.2%, R^2 went negative. The implementation has a bug where encoder memory is shared incorrectly across beam candidates -- the decoder's positional queries get misaligned when beams diverge to different sequence lengths. BLEU-4 stayed at 0.905, which is consistent with locally coherent but globally wrong outputs. Needs a proper batched-beam implementation.

**Transformer v2** with Noam warmup (d_model=256, 8 heads, 7.4M params): epoch 1 loss of 62.5, then stuck at ~2.66 for 50 epochs. Test exact match: 0%. The initial spike likely comes from the interaction between Noam LR and label smoothing at initialization. The subsequent plateau means the model found a degenerate solution early and never escaped. Would need larger warmup steps and probably no label smoothing to fix.

Both failures are documented in the notebook with analysis of what went wrong and what I'd try next.

---

## Setup

```bash
pip install torch sympy numpy scikit-learn nltk matplotlib
```

---

## References

- Lample & Charton (2019). Deep Learning for Symbolic Mathematics. [arxiv:1912.01412](https://arxiv.org/abs/1912.01412)
- Vaswani et al. (2017). Attention Is All You Need.
- Bahdanau et al. (2015). Neural Machine Translation by Jointly Learning to Align and Translate.
- ML4SCI FASEROH: [github.com/ML4SCI/FASEROH](https://github.com/ML4SCI/FASEROH)
