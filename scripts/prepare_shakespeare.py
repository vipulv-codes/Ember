"""Char-level Shakespeare -> .bin files for sanity ladder step 3."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    text = (DATA / "raw/shakespeare.txt").read_text()
    tok = CharTokenizer.from_text(text)
    tok.save(DATA / "shakespeare_chars.json")

    ids = np.array(tok.encode(text), dtype=np.uint16)
    split = int(0.9 * len(ids))
    ids[:split].tofile(DATA / "shakespeare_train.bin")
    ids[split:].tofile(DATA / "shakespeare_val.bin")

    meta = {"vocab_size": tok.vocab_size, "train_tokens": split, "val_tokens": len(ids) - split}
    (DATA / "shakespeare_meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
