# Axis: depth vs width at matched params (12x272 vs baseline 6x384).
run_name = "ablation_12m_deep"
preset = 'ember-12m-deep'
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
