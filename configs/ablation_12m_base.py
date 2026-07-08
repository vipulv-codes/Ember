# Ablation baseline: the Ember stack at 12M, 100M tokens, 16k tok/step.
# All variants below differ from this in exactly one axis.
run_name = "ablation_12m_base"
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
