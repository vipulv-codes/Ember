# Scaling study point 2/4. micro 16: micro 32 would put activations past the
# ~5GB WSL spill line at d=384/6 layers (extrapolated from measured 5M/30M).
run_name = "scale_12m"
preset = "ember-12m"
vocab_size = 8192
ctx_len = 512
total_tokens = 250_000_000
micro_bsz = 16
accum_steps = 32
peak_lr = 6e-4
warmup_frac = 0.02
eval_interval = 50
eval_batches = 32
ckpt_interval_sec = 900
