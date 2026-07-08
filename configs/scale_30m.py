# Scaling study point 3/4. Same micro/accum as the hero (measured 3.43GB,
# 42.6k tok/s cold); fresh run because the study needs the 250M budget with
# a fully annealed cosine, not a mid-schedule slice of the 600M hero run.
run_name = "scale_30m"
preset = "ember-30m"
vocab_size = 8192
ctx_len = 512
total_tokens = 250_000_000
micro_bsz = 16
accum_steps = 32
peak_lr = 3e-4
warmup_frac = 0.02
eval_interval = 50
eval_batches = 32
ckpt_interval_sec = 900
