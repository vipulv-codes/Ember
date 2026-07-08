# Sanity ladder step 3: ember-5M char-level on Shakespeare.
# Gate: samples must be recognizable English. Also the end-to-end shakedown
# of the training loop (checkpointing, resume, CSV logging).
run_name = "ladder3_shakespeare_char"
preset = "ember-5m"
vocab_size = 65
ctx_len = 256
train_bin = "data/shakespeare_train.bin"
val_bin = "data/shakespeare_val.bin"
total_tokens = 20_000_000  # ~20 epochs of the 1M-char corpus; overfitting is fine here
micro_bsz = 32
accum_steps = 1
peak_lr = 6e-4
eval_interval = 200
eval_batches = 16
ckpt_interval_sec = 45  # short run; keep well under total runtime so kill/resume is testable
