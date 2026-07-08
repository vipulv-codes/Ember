import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.mark.skipif(not (DATA / "meta.json").exists(), reason="run scripts/prepare_data.py first")
def test_bins_match_meta():
    meta = json.loads((DATA / "meta.json").read_text())
    for split in ["train", "val"]:
        arr = np.memmap(DATA / f"{split}.bin", dtype=np.uint16, mode="r")
        assert len(arr) == meta[f"{split}_tokens"]
        # spot-check token ids stay under the vocab on a sample
        sample = np.array(arr[: 1_000_000])
        assert sample.max() < meta["vocab_size"]
        # EOT separators must actually appear
        assert (sample == meta["eot_id"]).sum() > 0
