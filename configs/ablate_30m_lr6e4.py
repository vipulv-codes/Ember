# LR-sweep ablation, leg 1: scale_30m rerun with the small-model LR.
# Resolves the 30M kink in the scaling plot: if 6e-4 lands below the 12M's
# 1.5737, the kink was the LR policy, not the architecture or data.
run_name = "ablate_30m_lr6e4"
preset = "ember-30m"
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
