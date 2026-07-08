# LR-sweep ablation, leg 3: the 60M at the LR that fixed the 30M.
# With 30M@6e-4 at 1.4190, the 60M's 1.4696 (at 3e-4) is now the off-trend
# point. If 6e-4 helps here too, the flagship plot becomes LR-consistent at
# 6e-4 across all four sizes. Watch for instability: largest model, 2x LR.
run_name = "ablate_60m_lr6e4"
preset = "ember-60m"
vocab_size = 8192
ctx_len = 512
total_tokens = 250_000_000
micro_bsz = 8
accum_steps = 64
peak_lr = 6e-4
warmup_frac = 0.02
eval_interval = 50
eval_batches = 32
ckpt_interval_sec = 900
