"""Tokenize TinyStories-Instruct with the existing 8k BPE -> instruct .bins.

Records (Features/Words/Summary + Story) are <|endoftext|>-delimited like the
pretraining corpus, so the same streaming machinery applies unchanged.
"""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tokenizer import EOT, load_bpe

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from prepare_data import encode_split

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main():
    tok = load_bpe(DATA / "tokenizer.json")
    eot_id = tok.token_to_id(EOT)
    val = encode_split(tok, DATA / "raw/TinyStories-Instruct-valid.txt",
                       DATA / "instruct_val.bin", eot_id)
    train = encode_split(tok, DATA / "raw/TinyStories-Instruct-train.txt",
                         DATA / "instruct_train.bin", eot_id)
    meta = {"train_tokens": train, "val_tokens": val, "eot_id": eot_id}
    (DATA / "instruct_meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
