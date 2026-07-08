# LR-sweep ablation, leg 2 (reverse check): scale_12m rerun with the
# big-model LR. If 3e-4 pushes the 12M *above* the 30M's 1.5862, the LR
# story is symmetric and the confound is confirmed from both sides.
run_name = "ablate_12m_lr3e4"
preset = "ember-12m"
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
