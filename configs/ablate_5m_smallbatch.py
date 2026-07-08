# Ablation: batch size / critical-batch-size check (falsification run).
# Identical to scale_5m except 16k tokens/step (ladder-4 regime) instead of
# 262k -> ~15,258 steps instead of 953. Prediction if the code is sound and
# scale_5m's 1.864 was a step-starvation effect: final val < 1.754.
run_name = "ablate_5m_smallbatch"
preset = "ember-5m"
vocab_size = 8192
ctx_len = 512
total_tokens = 250_000_000
micro_bsz = 32
accum_steps = 1
peak_lr = 6e-4
warmup_frac = 0.02
eval_interval = 500
eval_batches = 32
ckpt_interval_sec = 900
