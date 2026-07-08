# The hero run: ember-30M, 600M tokens (~20 tokens/param, Chinchilla-style),
# one overnight run on the RTX 3060 Laptop 6GB.
run_name = "ember_30m_hero"
preset = "ember-30m"
vocab_size = 8192
ctx_len = 512
total_tokens = 600_000_000
# Measured on the 3060 6GB: micro 32 peaks at 6.30GB -> WDDM spills to shared
# memory, 10.8k tok/s. micro 16 peaks at 3.43GB -> 42.6k tok/s (4x faster).
micro_bsz = 16
accum_steps = 32  # 16*512*32 = 262,144 tokens/step -> ~2289 steps
peak_lr = 6e-4  # the LR sweep (experiments #13-14) showed 3e-4 costs ~0.15 nats at this scale
warmup_frac = 0.02
eval_interval = 25
eval_batches = 32
ckpt_interval_sec = 900
