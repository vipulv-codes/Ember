# Model card — ember-30M (and family)

A 29.9M-parameter decoder-only language model trained from scratch on
TinyStories, on a single RTX 3060 Laptop GPU (6GB). Part of a four-size
family (5M / 12M / 30M / 60M) built for a scaling study; **ember-30M is the
primary model**, with an instruction-tuned LoRA variant.

## Architecture

| | |
|---|---|
| Type | decoder-only transformer, causal LM |
| Params | 29,893,120 (hand-count formula asserted in tests) |
| Layers / heads / d_model | 8 / 8 / 512 |
| FFN | SwiGLU, hidden 1408, no biases anywhere |
| Norm | RMSNorm, pre-norm placement |
| Positions | RoPE (θ=10000), context 512 |
| Embeddings | weight-tied input/output |
| Precision | trained in bf16 autocast (fp32 master weights) |

Every architecture choice was ablated at 12M scale against its common
alternative (LayerNorm, post-norm, learned positions, GELU, deep-narrow);
each either wins its axis or ties it — see REPORT §5.

## Training

- **Data:** [TinyStoriesV2-GPT4](https://huggingface.co/datasets/roneneldan/TinyStories)
  — 536M train / 5.4M val tokens after tokenization with a custom
  byte-level BPE (vocab 8192) trained on the same corpus. Val split is the
  dataset's published valid file.
- **Recipe:** AdamW (β=0.9/0.95, wd 0.1, no decay on norms/embeddings),
  grad clip 1.0, 2% linear warmup → cosine to peak/10, peak LR 6e-4
  (sweep-validated; the standard size-based 3e-4 costs 0.088 nats here),
  262k tokens/step (micro 16 × accum 32 × ctx 512), 600M tokens
  (~20 tokens/param), seed 1337. One run, 5.24h.
- **Checkpoints:** `checkpoints/ember_30m_hero/best.pt` (val-selected).
  LoRA variant: `checkpoints/lora_30m_r8/` (534 KB adapter + merged model).

## Evaluation

| metric | value |
|---|---|
| val loss (TinyStories) | **1.2366** (ppl 3.44) |
| scaling fit across family (fixed 250M tokens, uniform LR) | L ∝ N^−0.139 |
| generation, bf16 GPU | ~100 tok/s single-stream |
| generation, int8 CPU + KV cache | 126 tok/s (quantization cost +0.049 nats) |

**LoRA instruct variant** (rank 8 on wq/wv, 0.44% trainable, 40M
TinyStories-Instruct tokens): instruct-format val loss 2.2366 → **1.4473**;
cost +0.09 on plain-story val. Blind dual-judge eval over 100 held-out
prompts: wins 90% of decisive matchups under gemini-2.5-flash (86/10/4) and
86% under gemini-2.5-flash-lite (86/14/0), 83% cross-judge agreement;
per-verdict JSON committed under `experiments/judge/`.

Qualitatively: fluent grammatical children's stories with arcs, dialogue,
and morals. Characteristic failure mode: referent confusion in
multi-character scenes.

## Intended use & limitations

Built as a learning/portfolio project: an end-to-end demonstration of
pretraining, scaling analysis, inference engineering, and fine-tuning at
laptop scale. Suitable for studying training dynamics and generating
TinyStories-style text. **Not suitable** for general-purpose generation,
factual tasks, or anything user-facing: the training distribution is
synthetic children's stories with a 512-token context and an 8k vocab.
Instruction-following in the LoRA variant is format-level, not
constraint-level. See REPORT §8 for the full limitations list.

## Reproduce

```bash
bash scripts/reproduce.sh all   # or: tests | data | ladder | hero | scaling | ablations | lora | judge
```
