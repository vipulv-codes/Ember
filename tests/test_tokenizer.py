from pathlib import Path

import pytest

from tokenizer import EOT, CharTokenizer, load_bpe

ROOT = Path(__file__).resolve().parents[1]
BPE_PATH = ROOT / "data" / "tokenizer.json"


def test_char_round_trip(tmp_path):
    text = "To be, or not to be: that is the question.\n"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text
    tok.save(tmp_path / "chars.json")
    tok2 = CharTokenizer.load(tmp_path / "chars.json")
    assert tok2.decode(tok2.encode(text)) == text


@pytest.mark.skipif(not BPE_PATH.exists(), reason="run scripts/prepare_data.py first")
def test_bpe_round_trip():
    tok = load_bpe(BPE_PATH)
    samples = [
        "Once upon a time, there was a little girl named Lily.",
        'She said, "Let\'s go to the park!" and they all cheered.',
        "Numbers like 3 and 7, punctuation; even\nnewlines survive.",
    ]
    for s in samples:
        assert tok.decode(tok.encode(s).ids) == s


@pytest.mark.skipif(not BPE_PATH.exists(), reason="run scripts/prepare_data.py first")
def test_bpe_vocab_and_special():
    tok = load_bpe(BPE_PATH)
    assert tok.get_vocab_size() == 8192
    assert tok.token_to_id(EOT) is not None
