# Sanity ladder step 4: ember-5M on a TinyStories subset (fixed token budget).
# Gate: coherent-ish sentences.
run_name = "ladder4_tinystories_5m"
preset = "ember-5m"
vocab_size = 8192
ctx_len = 512
total_tokens = 60_000_000
micro_bsz = 32
accum_steps = 1
peak_lr = 6e-4
eval_interval = 300
eval_batches = 16
ckpt_interval_sec = 300
