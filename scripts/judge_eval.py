"""LLM-judge evaluation (plan sections 3C & 5): win-rate between two
checkpoints on held-out TinyStories-Instruct prompts, scored by Gemini on a
grammar / coherence / creativity / instruction-following rubric.

Judge sees the two completions in randomized A/B order (position-bias guard);
per-prompt JSON verdicts land in experiments/judge/ so scores are auditable.

Usage:
    python scripts/judge_eval.py --ckpt-a checkpoints/ember_30m_hero/best.pt \
        --ckpt-b checkpoints/lora_30m_r8/merged.pt --n 100
Requires GEMINI_API_KEY in .env or the environment.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import httpx
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sample import load_model
from tokenizer import EOT, iter_stories, load_bpe

ROOT = Path(__file__).resolve().parents[1]
JUDGE_MODEL = "gemini-2.5-flash-lite"  # free-tier daily quota actually allows a 100-prompt eval

RUBRIC = """You are judging two children's stories written by two different small language models from the same writing instructions.

INSTRUCTIONS GIVEN TO BOTH MODELS:
{prompt}

STORY A:
{story_a}

STORY B:
{story_b}

Score each story 1-10 on: grammar, coherence, creativity, and how well it follows the instructions. Then pick the overall winner.
Respond with ONLY this JSON (no markdown fences):
{{"a": {{"grammar": n, "coherence": n, "creativity": n, "instruction": n}}, "b": {{"grammar": n, "coherence": n, "creativity": n, "instruction": n}}, "winner": "A" or "B" or "tie", "reason": "one sentence"}}"""


def api_key() -> str:
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    sys.exit("GEMINI_API_KEY not found (env or .env)")


# Pinned judge decoding config — never rely on server-side defaults.
# thinkingBudget 0: explicit no-thinking for BOTH judges (flash defaults to
# dynamic thinking, flash-lite to none; pinning makes them uniform, cheap,
# and immune to degenerate-input runaway generation). maxOutputTokens is
# safe to cap only because thinking is off (thinking tokens would consume
# the cap and yield empty responses).
GEN_CONFIG = {
    "temperature": 0.0,
    "responseMimeType": "application/json",
    "maxOutputTokens": 1024,
    "thinkingConfig": {"thinkingBudget": 0},
}


def judge(client: httpx.Client, key: str, text: str, model: str, retries: int = 8):
    """Returns (verdict_dict, usage_metadata)."""
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{model}:generateContent?key={key}")
    body = {"contents": [{"parts": [{"text": text}]}], "generationConfig": GEN_CONFIG}
    for attempt in range(retries):
        try:
            resp = client.post(url, json=body, timeout=60)
        except httpx.HTTPError:
            time.sleep(20)
            continue
        if resp.status_code in (429, 500, 503):
            time.sleep(30 * (attempt + 1))
            continue
        resp.raise_for_status()
        data = resp.json()
        parts = data["candidates"][0]["content"]["parts"]
        raw = "".join(p.get("text", "") for p in parts).strip()
        if raw.startswith("```"):
            raw = raw.strip("`").removeprefix("json").strip()
        try:
            # tolerate trailing junk after the JSON object (flash-lite does this)
            obj, _ = json.JSONDecoder().raw_decode(raw)
            return obj, data.get("usageMetadata", {})
        except json.JSONDecodeError:
            continue  # malformed verdict: re-ask
    raise RuntimeError("judge kept failing (rate limit or malformed output)")


def held_out_prompts(n: int, max_prompt_chars: int = 600) -> list[str]:
    """Instruction blocks (everything before 'Story:') from the valid split."""
    prompts = []
    for record in iter_stories(ROOT / "data/raw/TinyStories-Instruct-valid.txt"):
        if record.count("Story:") == 1:
            head = record.split("Story:")[0].strip()
            if 0 < len(head) <= max_prompt_chars and "Summary:" in head:
                prompts.append(head + "\nStory:")
        if len(prompts) >= n:
            break
    return prompts


@torch.no_grad()
def complete(model, tok, prompt: str, device: str, seed: int) -> str:
    torch.manual_seed(seed)
    idx = torch.tensor([tok.encode(prompt).ids], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=300, temperature=0.8, top_p=0.95,
                         eos_id=tok.token_to_id(EOT))
    text = tok.decode(out[0].tolist())
    return text[len(prompt):].replace(EOT, "").strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", required=True, help="baseline (counts as 'A' in results)")
    ap.add_argument("--ckpt-b", required=True, help="challenger")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--name", default=None, help="output file stem")
    ap.add_argument("--judge-model", default=JUDGE_MODEL)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = load_bpe(ROOT / "data/tokenizer.json")
    model_a, cfg_a, _ = load_model(args.ckpt_a, device)
    model_b, cfg_b, _ = load_model(args.ckpt_b, device)
    name = args.name or f"{cfg_a['run_name']}_vs_{cfg_b['run_name']}"
    out_dir = ROOT / "experiments" / "judge"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.jsonl"

    prompts = held_out_prompts(args.n)
    print(f"{len(prompts)} held-out prompts; judge = {args.judge_model}; writing {out_path}",
          flush=True)

    key = api_key()
    client = httpx.Client()
    rng = random.Random(1337)
    wins = {"a": 0, "b": 0, "tie": 0}
    done = 0
    if out_path.exists():  # resume: trust the JSONL, skip what's scored
        for line in out_path.read_text().splitlines():
            rec = json.loads(line)
            wins[rec["winner_real"]] += 1
            done += 1
        print(f"resuming: {done} verdicts already on disk "
              f"(a={wins['a']} b={wins['b']} tie={wins['tie']})")
    records = []
    for i, prompt in enumerate(prompts):
        flip_check = rng.random() < 0.5  # keep RNG stream aligned with a fresh run
        if i < done:
            continue
        gen_a = complete(model_a, tok, prompt, device, seed=1337 + i)
        gen_b = complete(model_b, tok, prompt, device, seed=1337 + i)
        flip = flip_check  # drawn once per prompt above (resume-safe stream)
        first, second = (gen_b, gen_a) if flip else (gen_a, gen_b)
        verdict, usage = judge(client, key,
                               RUBRIC.format(prompt=prompt, story_a=first, story_b=second),
                               model=args.judge_model)
        w = verdict.get("winner", "tie").upper()
        real = "tie" if w == "TIE" else ("b" if (w == "A") == flip else "a")
        wins[real] += 1
        rec = {"i": i, "prompt": prompt, "gen_a": gen_a, "gen_b": gen_b,
               "flipped": flip, "verdict": verdict, "winner_real": real,
               "usage": {k: usage.get(k, 0) for k in
                         ["promptTokenCount", "candidatesTokenCount", "thoughtsTokenCount",
                          "totalTokenCount"]}}
        records.append(rec)
        with open(out_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"[{i + 1}/{len(prompts)}] a={wins['a']} b={wins['b']} tie={wins['tie']}",
              flush=True)
        time.sleep(1)

    n_scored = sum(wins.values())
    tok_in = sum(r["usage"]["promptTokenCount"] for r in records)
    tok_out = sum(r["usage"]["candidatesTokenCount"] + r["usage"]["thoughtsTokenCount"]
                  for r in records)
    print(f"token usage this session: {tok_in:,} in / {tok_out:,} out "
          f"(thinking tokens: {sum(r['usage']['thoughtsTokenCount'] for r in records):,})")
    summary = {
        "judge": args.judge_model, "n": n_scored,
        "generation_config": GEN_CONFIG,
        "model_a": cfg_a["run_name"], "model_b": cfg_b["run_name"],
        "wins_a": wins["a"], "wins_b": wins["b"], "ties": wins["tie"],
        "winrate_b_vs_a": round(wins["b"] / max(1, n_scored - wins["tie"]), 3),
    }
    (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
