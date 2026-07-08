"""LoRA-instruction-tune ember-30M on TinyStories-Instruct (plan section 5 leg 1).

Freezes the pretrained model, adapts wq/wv with rank-r LoRA, trains on the
instruct .bins, and reports instruct-val loss before/after plus pretrain-val
regression. Saves the tiny adapter and a merged full checkpoint (the merged
file is sample.py-compatible).

Usage: python scripts/finetune_lora.py [--r 8] [--alpha 16] [--tokens 40000000]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lora import apply_lora, lora_state_dict, merge_lora
from sample import load_model
from train import TrainConfig, append_csv, evaluate, get_batch, lr_at

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/ember_30m_hero/best.pt")
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--tokens", type=int, default=40_000_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, base_cfg, _ = load_model(args.ckpt, device)

    cfg = TrainConfig(
        run_name=f"lora_30m_r{args.r}",
        preset=base_cfg["preset"],
        train_bin="data/instruct_train.bin",
        val_bin="data/instruct_val.bin",
        total_tokens=args.tokens,
        micro_bsz=16,
        accum_steps=2,
        peak_lr=args.lr,
        eval_interval=300,
        eval_batches=32,
        device=device,
    )
    train_data = np.memmap(ROOT / cfg.train_bin, dtype=np.uint16, mode="r")
    val_data = np.memmap(ROOT / cfg.val_bin, dtype=np.uint16, mode="r")
    pretrain_val = np.memmap(ROOT / "data/val.bin", dtype=np.uint16, mode="r")

    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    before_instruct = evaluate(model, val_data, cfg, autocast)
    before_pretrain = evaluate(model, pretrain_val, cfg, autocast)
    print(f"BASE:  instruct-val {before_instruct:.4f}  pretrain-val {before_pretrain:.4f}")

    n_trainable = apply_lora(model, r=args.r, alpha=args.alpha)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA r={args.r} alpha={args.alpha}: {n_trainable:,} trainable "
          f"({100 * n_trainable / total:.2f}% of {total / 1e6:.1f}M)")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.peak_lr, betas=(0.9, 0.95), weight_decay=0.0)
    csv_path = ROOT / "experiments" / "runs" / f"{cfg.run_name}.csv"
    csv_path.unlink(missing_ok=True)

    model.train()
    t0 = time.time()
    for step in range(cfg.total_steps):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        for micro in range(cfg.accum_steps):
            x, y = get_batch(train_data, cfg, step, micro, split_salt=0)
            with autocast:
                _, loss = model(x, targets=y)
            (loss / cfg.accum_steps).backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % 50 == 0 or step == cfg.total_steps - 1:
            row = {"step": step, "tokens": (step + 1) * cfg.tokens_per_step,
                   "lr": f"{lr:.2e}", "train_loss": f"{loss.item():.4f}", "val_loss": ""}
            if (step % cfg.eval_interval == 0 and step > 0) or step == cfg.total_steps - 1:
                v = evaluate(model, val_data, cfg, autocast)
                row["val_loss"] = f"{v:.4f}"
                print(f"step {step:5d}/{cfg.total_steps} loss {row['train_loss']} "
                      f"instruct-val {v:.4f}", flush=True)
            else:
                print(f"step {step:5d}/{cfg.total_steps} loss {row['train_loss']}", flush=True)
            append_csv(csv_path, row)

    after_instruct = evaluate(model, val_data, cfg, autocast)
    after_pretrain = evaluate(model, pretrain_val, cfg, autocast)
    print(f"\nBASE:  instruct-val {before_instruct:.4f}  pretrain-val {before_pretrain:.4f}")
    print(f"LoRA:  instruct-val {after_instruct:.4f}  pretrain-val {after_pretrain:.4f}")
    print(f"delta: instruct {after_instruct - before_instruct:+.4f}  "
          f"pretrain {after_pretrain - before_pretrain:+.4f}  "
          f"({(time.time() - t0) / 60:.0f} min)")

    out_dir = ROOT / "checkpoints" / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"adapter": lora_state_dict(model), "r": args.r, "alpha": args.alpha,
                "targets": ["wq", "wv"], "base_ckpt": args.ckpt}, out_dir / "adapter.pt")
    merged = merge_lora(model)
    torch.save({"model": merged.state_dict(), "step": cfg.total_steps,
                "config": {**base_cfg, "run_name": cfg.run_name}},
               out_dir / "merged.pt")
    print(f"saved {out_dir}/adapter.pt "
          f"({(out_dir / 'adapter.pt').stat().st_size / 1e3:.0f} KB) and merged.pt")


if __name__ == "__main__":
    main()
