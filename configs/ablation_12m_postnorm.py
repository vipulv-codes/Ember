# Axis: pre-norm vs post-norm (original transformer placement).
# Expected to train worse or diverge - that IS the result if so.
run_name = "ablation_12m_postnorm"
norm_placement = "post"
preset = 'ember-12m'
vocab_size = 8192
ctx_len = 512
total_tokens = 100_000_000
micro_bsz = 16
accum_steps = 2
peak_lr = 6e-4
warmup_frac = 0.02
eval_interval = 500
eval_batches = 32
ckpt_interval_sec = 900
