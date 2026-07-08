# Scaling study point 4/4. micro 8: measured 3.01GB / 23.6k tok/s vs
# micro 16's 4.97GB / 26.9k — 12% slower for 2GB of headroom on an
# unattended multi-hour run. Worth it.
run_name = "scale_60m"
preset = "ember-60m"
vocab_size = 8192
ctx_len = 512
total_tokens = 250_000_000
micro_bsz = 8
accum_steps = 64
peak_lr = 3e-4
warmup_frac = 0.02
eval_interval = 50
eval_batches = 32
ckpt_interval_sec = 900
