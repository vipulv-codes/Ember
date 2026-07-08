# Scaling study point 1/4 (plan section 3A): fixed 250M-token budget,
# matched 262k tokens/step effective batch across all four sizes.
run_name = "scale_5m"
preset = "ember-5m"
vocab_size = 8192
ctx_len = 512
total_tokens = 250_000_000
micro_bsz = 32
accum_steps = 16
peak_lr = 6e-4
warmup_frac = 0.02
eval_interval = 50
eval_batches = 32
ckpt_interval_sec = 900
