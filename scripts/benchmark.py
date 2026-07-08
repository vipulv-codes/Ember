"""Inference benchmarks (plan section 4, milestone M3).

Measures generation tokens/sec for ember-30M across:
  fp32 CPU / int8-dynamic-quant CPU (torch.ao) / bf16 GPU  x  KV cache on/off
plus the KV-cache memory and the quantization quality cost (val loss fp32 vs
int8 on fixed batches).

Writes experiments/benchmarks.csv; chart via scripts/plot_benchmarks.py.
"""

import argparse
import copy
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model import KVCache
from sample import load_model
from tokenizer import load_bpe

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Once upon a time, there was a little girl named Lily."
MAX_NEW = 128


def timed_generate(model, idx, use_cache, ctx=nullcontext(), sync=None) -> float:
    """Greedy-generate MAX_NEW tokens; returns tokens/sec (after 1 warmup)."""
    for _ in range(2):
        with ctx:
            model.generate(idx, max_new_tokens=8, temperature=0.0, use_cache=use_cache)
        if sync:
            sync()
        t0 = time.perf_counter()
        with ctx:
            out = model.generate(
                idx, max_new_tokens=MAX_NEW, temperature=0.0, use_cache=use_cache
            )
        if sync:
            sync()
        dt = time.perf_counter() - t0
    assert out.shape[1] == idx.shape[1] + MAX_NEW
    return MAX_NEW / dt


@torch.no_grad()
def val_loss(model, device, dtype_ctx, batches=12, bsz=8, ctx_len=512, seed=1337):
    data = np.memmap(ROOT / "data" / "val.bin", dtype=np.uint16, mode="r")
    g = torch.Generator().manual_seed(seed)
    losses = []
    for _ in range(batches):
        ix = torch.randint(0, len(data) - ctx_len - 1, (bsz,), generator=g)
        x = torch.stack([torch.from_numpy(data[i : i + ctx_len].astype(np.int64)) for i in ix])
        y = torch.stack(
            [torch.from_numpy(data[i + 1 : i + 1 + ctx_len].astype(np.int64)) for i in ix]
        )
        with dtype_ctx:
            _, loss = model(x.to(device), targets=y.to(device))
        losses.append(loss.item())
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ember_30m_hero/best.pt")
    args = ap.parse_args()

    tok = load_bpe(ROOT / "data" / "tokenizer.json")
    prompt_ids = tok.encode(PROMPT).ids
    rows = []

    # --- CPU fp32 ---
    model, cfg, _ = load_model(args.ckpt, "cpu")
    model = model.float().eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long)
    for use_cache in [False, True]:
        tps = timed_generate(model, idx, use_cache)
        rows.append({"variant": "fp32 CPU", "kv_cache": use_cache, "tok_per_sec": round(tps, 1)})
        print(rows[-1], flush=True)
    loss_fp32 = val_loss(model, "cpu", nullcontext())

    # --- CPU int8 dynamic quant (torch.ao) ---
    qmodel = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model), {torch.nn.Linear}, dtype=torch.qint8
    )
    for use_cache in [False, True]:
        tps = timed_generate(qmodel, idx, use_cache)
        rows.append({"variant": "int8 CPU", "kv_cache": use_cache, "tok_per_sec": round(tps, 1)})
        print(rows[-1], flush=True)
    loss_int8 = val_loss(qmodel, "cpu", nullcontext())

    # --- GPU bf16 ---
    if torch.cuda.is_available():
        model_gpu, _, _ = load_model(args.ckpt, "cuda")
        model_gpu.eval()
        idx_gpu = idx.to("cuda")
        bf16 = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        for use_cache in [False, True]:
            tps = timed_generate(
                model_gpu, idx_gpu, use_cache, ctx=bf16, sync=torch.cuda.synchronize
            )
            rows.append(
                {"variant": "bf16 GPU", "kv_cache": use_cache, "tok_per_sec": round(tps, 1)}
            )
            print(rows[-1], flush=True)

        # KV-cache memory: formula vs measured
        c = model_gpu.cfg
        cache = KVCache(c, 1)
        with bf16, torch.no_grad():
            model_gpu(idx_gpu, cache=cache)
        formula = 2 * c.n_layers * c.n_heads * c.head_dim * c.ctx_len * 2  # bf16 bytes
        print(f"KV cache @ ctx {c.ctx_len}: measured {cache.nbytes / 1e6:.2f} MB, "
              f"formula {formula / 1e6:.2f} MB")

    ppl = lambda L: float(np.exp(L))
    quality = {
        "val_loss_fp32": round(loss_fp32, 4),
        "val_loss_int8": round(loss_int8, 4),
        "ppl_fp32": round(ppl(loss_fp32), 3),
        "ppl_int8": round(ppl(loss_int8), 3),
    }
    print(json.dumps(quality, indent=2))

    out = ROOT / "experiments" / "benchmarks.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["variant", "kv_cache", "tok_per_sec"])
        w.writeheader()
        w.writerows(rows)
    (ROOT / "experiments" / "benchmark_quality.json").write_text(json.dumps(quality, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
