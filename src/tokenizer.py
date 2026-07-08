"""Tokenizers for Ember.

- CharTokenizer: char-level codec for the Shakespeare sanity-ladder runs.
- BPE (HF `tokenizers`): byte-level BPE, vocab 8192, trained on TinyStories.
  Stories in the raw txt files are delimited by lines of `<|endoftext|>`;
  that string is the single special token and doubles as the document separator.
"""

import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

EOT = "<|endoftext|>"


class CharTokenizer:
    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {i: c for c, i in stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        chars = sorted(set(text))
        return cls({c: i for i, c in enumerate(chars)})

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str | Path):
        Path(path).write_text(json.dumps(self.stoi))

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        return cls(json.loads(Path(path).read_text()))


def iter_stories(txt_path: str | Path):
    """Yield individual stories from a TinyStories txt file (EOT-delimited)."""
    buf: list[str] = []
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            if line.strip() == EOT:
                story = "".join(buf).strip()
                if story:
                    yield story
                buf = []
            else:
                buf.append(line)
    story = "".join(buf).strip()
    if story:
        yield story


def train_bpe(story_iter, vocab_size: int, out_path: str | Path) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        show_progress=True,
    )
    tokenizer.train_from_iterator(story_iter, trainer=trainer)
    tokenizer.save(str(out_path))
    return tokenizer


def load_bpe(path: str | Path) -> Tokenizer:
    return Tokenizer.from_file(str(path))
