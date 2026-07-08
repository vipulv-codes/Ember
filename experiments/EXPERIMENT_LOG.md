# Experiment log

Every run gets an entry: config, commit, what changed, result, artifacts. Fixed seed policy: seed=1337 unless the run *is* a seed ablation.

| # | Run | Config | Result | Notes |
|---|-----|--------|--------|-------|
| 1 | ladder1 bigram | scripts/bigram_baseline.py | PASS: loss 2.475 vs empirical bigram entropy floor 2.453 nats | char bigram on Shakespeare; sample is expected bigram gibberish |
| 2 | ladder2 overfit-one-batch | ember-5m char, ctx 256, 8x256 batch, lr 1e-3 | PASS: loss 0.0007 after 500 steps (11s, GPU) | classic bug detector; masking/LR/init all sane |
| 3 | tokenizer + data | BPE vocab 8192, byte-level, trained on every 4th story | 536.1M train / 5.4M val tokens, 4.16 chars/token | TinyStoriesV2-GPT4; EOT id 0; uint16 bins |
| 4 | ladder3 char 5M | configs/ladder3_shakespeare_char.py | PASS: recognizable English; best val 1.539 @ step 801, train 0.84 @ end (overfit as expected, ~20 epochs) | ~185k tok/s, 0.62GB VRAM; killed at step 1120 + `--resume` from 1039: seamless, no loss spike |
| 5 | ladder4 BPE 5M | configs/ladder4_tinystories_5m.py | PASS: coherent stories with arc + moral; val 1.754 (still falling) | 60M tokens in 7 min (~140k tok/s), 2.95GB |
| 6 | 30M throughput bench | hero cfg, 10 steps | micro 32: 6.30GB (spills past 6GB VRAM) -> 10.8k tok/s. micro 16 x accum 32: 3.43GB -> 42.6k tok/s (4x) | WDDM shared-memory spill is the silent killer; keep peak under ~5GB |
| 7 | **ember_30m_hero** | configs/ember_30m_hero.py | **DONE: best val 1.3245 (ppl 3.76)**, 600M tokens in 6.86h; fluent stories, referent slips in multi-character scenes | throttled to 14k tok/s @ 88C overnight; survived a Windows sleep (WSL VM freeze) mid-run; gallery + png in repo |
| 8 | scale_5m | configs/scale_5m.py | val 1.8644 @ 250M tokens | worse than ladder4's 1.754 @ 60M — 262k batch step-starves a 5M model; triggered ablation #11 |
| 9 | scale_12m | configs/scale_12m.py | val 1.5737 @ 250M tokens | ~20 tok/param, the most Chinchilla-comfortable point |
| 10 | scale_30m + scale_60m | configs/scale_{30m,60m}.py | 30M: 1.5862 (above trend); 60M: **1.4696** | fit L ∝ N^-0.093; 30M kink likely the 3e-4-vs-6e-4 LR policy confound — LR sweep will resolve; report/scaling.png |
| 11 | ablate_5m_smallbatch | configs/ablate_5m_smallbatch.py | **CONFIRMED: val 1.4856** (predicted < 1.754); 0.38 nats better than #8 from batch size alone | 15,258 steps vs 953 at same tokens/LR; small-batch 5M nearly ties giant-batch 60M — batch regime is a first-order variable |
| 12 | KV cache + benchmarks (M3) | scripts/benchmark.py | cache: 2.8x fp32 CPU, 2.6x int8 CPU, 0.9x GPU (launch-bound); int8+cache CPU 146 tok/s > bf16 GPU 106; int8 ppl 3.71->3.79 | cache 8.39MB @ ctx512 == formula; 5 correctness tests; report/benchmarks.png |
| 13 | LR sweep (30M @ 6e-4, 12M @ 3e-4) | configs/ablate_{30m_lr6e4,12m_lr3e4}.py | **BOTH CONFIRMED: 30M@6e-4 = 1.4190** (was 1.5862 @ 3e-4; predicted < 1.5737); **12M@3e-4 = 1.7906** (was 1.5737 @ 6e-4; predicted > 1.5862) | scaling-plot kink was the LR policy, symmetric on both sides; ~0.2 nats per 2x LR at this 953-step regime |
| 14 | ablate_60m_lr6e4 | configs/ablate_60m_lr6e4.py | **val 1.3530** (was 1.4696 @ 3e-4); no instability | completes the LR-consistent flagship plot: 1.8644 / 1.5737 / 1.4190 / 1.3530, fit L ∝ N^-0.139 |
| 15 | 6-run ablation set @ 12M, 100M tokens, 16k batch | configs/ablation_12m_*.py | base **1.4525**; layernorm 1.4527 (tie); deep-12x272 1.4618 (+.009, and 1.6x slower wall-clock); gelu 1.4659 (+.013); learnedpos 1.4748 (+.022); postnorm 1.4904 (+.038, stable but worse) | every modern-stack choice validated or tied; RMSNorm = quality tie, wins on compute |
| 16 | LoRA r=8 on hero wq/wv (M4) | scripts/finetune_lora.py | instruct-val 2.3452→**1.5534** (−0.79); pretrain-val +0.10 tax; 131k trainable (0.44%), 534KB adapter, 20 min | merged ckpt at checkpoints/lora_30m_r8/ |
| 17 | judge eval, dual-judge n=100 | scripts/judge_eval.py | LoRA **80/13/7** (gemini-2.5-flash, 86% of decisive), **81/19/0** (flash-lite, 81%); 73% exact agreement | pinned judge config (temp 0, JSON, no thinking, capped output) recorded in summaries; per-verdict token accounting; rubric improves on all 4 dimensions under both judges |
| 18 | **ember_30m_hero @ 6e-4** (v2) | configs/ember_30m_hero.py | **val 1.2366 (ppl 3.44)**, 600M tokens in 5.24h; was 1.3245 @ 3e-4 — the #13 LR finding transfers to the 2288-step schedule (−0.088 nats) | closes the review question "would the hero improve at 6e-4"; gallery/curve regenerated |
| 19 | benchmarks + LoRA v2 on hero v2 | scripts/{benchmark,finetune_lora}.py | cache 3.1×/2.4× CPU, 1.1× GPU (wash confirmed); int8 +0.049 nats (sharper model pays more); LoRA instruct 2.2366→**1.4473**, tax +0.09 | same harness, new checkpoint |
| 20 | judge eval v2, dual-judge n=100 | scripts/judge_eval.py | LoRA **86/10/4** (flash, 90% of decisive), **86/14/0** (flash-lite, 86%); **83%** exact agreement | strongest eval yet; fresh verdicts, same pinned config |
| 21 | seed variance @ 5M (seeds 1338/1339) | scale_5m + run_name/seed overrides | 1.8545 / 1.8583 vs 1.8644 (s1337): **σ = 0.0050**, range 0.0099 | quantifies the §8 single-seed limitation |
| 22 | seed variance @ 12M (seeds 1338/1339) | scale_12m + overrides | 1.5790 / 1.5838 vs 1.5737 (s1337): **σ = 0.0051**, range 0.0101 | scaling ordering ≥13σ clear of seed noise; error bars added to report/scaling.png |
