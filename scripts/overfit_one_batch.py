"""Sanity ladder step 2: ember-5M must overfit a single fixed batch to ~0 loss.
The classic bug detector — catches masking, off-by-one, LR, and init mistakes.
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import Ember, preset_config
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = (ROOT / "data/raw/shakespeare.txt").read_text()
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)

    cfg = preset_config("ember-5m", vocab_size=tok.vocab_size, ctx_len=256)
    model = Ember(cfg).to(device)
    print(f"ember-5m (char): {cfg.n_params():,} params, device {device}")

    # One fixed batch, 8 x 256 tokens.
    ix = torch.randint(0, len(data) - cfg.ctx_len - 1, (8,))
    x = torch.stack([data[i : i + cfg.ctx_len] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + cfg.ctx_len + 1] for i in ix]).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95))
    t0 = time.time()
    for step in range(500):
        _, loss = model(x, targets=y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 499:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    final = loss.item()
    print(f"\nfinal loss {final:.4f} after 500 steps ({time.time() - t0:.0f}s)")
    assert final < 0.1, f"FAILED to overfit one batch: loss {final:.4f} >= 0.1"
    print("LADDER 2 PASS: single batch overfit to ~0 loss")


if __name__ == "__main__":
    main()
