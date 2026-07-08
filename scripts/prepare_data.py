"""Tokenize TinyStories once -> memmapped uint16 .bin files.

1. Train byte-level BPE (vocab 8192) on a story subsample (every Nth story is
   plenty to learn 8k merges, and much faster than the full 2.1GB).
2. Encode the full train/valid corpora, appending <|endoftext|> after each
   story, streamed in chunks so RAM stays flat on a 7.5GB WSL box.

Outputs: data/tokenizer.json, data/train.bin, data/val.bin, data/meta.json
"""

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tokenizer import EOT, iter_stories, load_bpe, train_bpe

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VOCAB_SIZE = 8192
TRAIN_TXT = DATA / "raw/TinyStoriesV2-GPT4-train.txt"
VAL_TXT = DATA / "raw/TinyStoriesV2-GPT4-valid.txt"
TOKENIZER_SUBSAMPLE = 4  # train BPE on every 4th story
ENCODE_CHUNK = 50_000  # stories per encode_batch call


def get_tokenizer():
    path = DATA / "tokenizer.json"
    if path.exists():
        print(f"tokenizer exists: {path}")
        return load_bpe(path)
    print(f"training BPE vocab={VOCAB_SIZE} on every {TOKENIZER_SUBSAMPLE}th story ...")
    t0 = time.time()
    subsample = itertools.islice(iter_stories(TRAIN_TXT), 0, None, TOKENIZER_SUBSAMPLE)
    tok = train_bpe(subsample, VOCAB_SIZE, path)
    print(f"tokenizer trained in {time.time() - t0:.0f}s -> {path}")
    return tok


def encode_split(tok, txt_path: Path, out_path: Path, eot_id: int) -> int:
    t0 = time.time()
    total = 0
    stories = iter_stories(txt_path)
    with open(out_path, "wb") as f:
        while True:
            chunk = list(itertools.islice(stories, ENCODE_CHUNK))
            if not chunk:
                break
            encs = tok.encode_batch(chunk)
            ids: list[int] = []
            for e in encs:
                ids.extend(e.ids)
                ids.append(eot_id)
            arr = np.array(ids, dtype=np.uint16)
            assert arr.max() < VOCAB_SIZE
            arr.tofile(f)
            total += len(arr)
            print(f"  {out_path.name}: {total / 1e6:8.1f}M tokens ({time.time() - t0:.0f}s)", flush=True)
    return total


def main():
    tok = get_tokenizer()
    eot_id = tok.token_to_id(EOT)
    assert eot_id is not None

    val_tokens = encode_split(tok, VAL_TXT, DATA / "val.bin", eot_id)
    train_tokens = encode_split(tok, TRAIN_TXT, DATA / "train.bin", eot_id)

    meta = {
        "vocab_size": VOCAB_SIZE,
        "eot_id": eot_id,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "dtype": "uint16",
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"chars/token ~ {TRAIN_TXT.stat().st_size / train_tokens:.2f}")


if __name__ == "__main__":
    main()
