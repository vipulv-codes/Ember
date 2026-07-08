"""Sanity ladder step 1: a char-level bigram model must train and overfit
Shakespeare. If this doesn't work, nothing downstream will.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]


def main():
    torch.manual_seed(1337)
    text = (ROOT / "data/raw/shakespeare.txt").read_text()
    tok = CharTokenizer.from_text(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    print(f"corpus: {len(data):,} chars, vocab {tok.vocab_size}")

    # Bigram model: a (V, V) table of next-char logits.
    logits_table = nn.Embedding(tok.vocab_size, tok.vocab_size)
    opt = torch.optim.AdamW(logits_table.parameters(), lr=1e-1)

    # Empirical bigram entropy = the floor this model should approach.
    counts = torch.zeros(tok.vocab_size, tok.vocab_size)
    counts.index_put_((data[:-1], data[1:]), torch.ones(len(data) - 1), accumulate=True)
    probs = (counts + 1e-9) / (counts + 1e-9).sum(1, keepdim=True)
    row_entropy = -(probs * probs.log()).sum(1)
    floor = (row_entropy * counts.sum(1) / counts.sum()).sum()
    print(f"empirical bigram entropy floor: {floor:.3f} nats")

    for step in range(301):
        ix = torch.randint(0, len(data) - 1, (4096,))
        loss = F.cross_entropy(logits_table(data[ix]), data[ix + 1])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.3f}")

    assert loss.item() < floor + 0.1, f"bigram failed to reach entropy floor {floor:.3f}"
    ids = [tok.encode("T")[0]]
    for _ in range(200):
        p = logits_table(torch.tensor([ids[-1]])).softmax(-1)
        ids.append(torch.multinomial(p, 1).item())
    print("\nsample:\n" + tok.decode(ids))
    print("\nLADDER 1 PASS: bigram reached its entropy floor")


if __name__ == "__main__":
    main()
