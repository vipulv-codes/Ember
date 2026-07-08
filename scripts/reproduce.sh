#!/usr/bin/env bash
# Reproduce Ember from an empty checkout, stage by stage.
#
#   bash scripts/reproduce.sh [stage]
#
#   tests      fast CPU test suite (~10s)
#   data       download + tokenize TinyStories and Shakespeare (~15 min, ~5GB disk)
#   ladder     sanity ladder 1-4: bigram, overfit-one-batch, char 5M, BPE 5M (~15 min GPU)
#   hero       ember-30M, 600M tokens (~4-7h GPU depending on thermals)
#   gallery    30-prompt sample gallery + sampling table + loss curve for the hero
#   scaling    4 sizes x 250M tokens at uniform LR 6e-4 (~10h GPU) + flagship plot
#   ablations  6 axes at 12M x 100M tokens (~4h GPU)
#   bench      KV-cache / precision benchmarks + chart (~10 min, needs idle GPU)
#   lora       LoRA instruct-tune the hero + before/after eval (~25 min GPU)
#   judge      dual-judge eval, 100 prompts (needs GEMINI_API_KEY in .env; ~$0.05)
#   all        everything above, in order
#
# Every training stage is resumable: rerun the same command after a crash and
# it continues from the newest checkpoint (add --resume where noted below if
# invoking train.py by hand).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
stage="${1:-all}"

run() { echo; echo "=== $* ==="; "$@"; }

do_tests()   { run $PY -m pytest -q; }
do_data()    { run $PY scripts/prepare_data.py
               run $PY scripts/prepare_shakespeare.py
               run $PY scripts/prepare_instruct.py; }
do_ladder()  { run $PY scripts/bigram_baseline.py
               run $PY scripts/overfit_one_batch.py
               run $PY src/train.py --config configs/ladder3_shakespeare_char.py
               run $PY src/train.py --config configs/ladder4_tinystories_5m.py; }
do_hero()    { run $PY src/train.py --config configs/ember_30m_hero.py; }
do_gallery() { run $PY scripts/sample_gallery.py --ckpt checkpoints/ember_30m_hero/best.pt
               run $PY scripts/sampling_table.py --ckpt checkpoints/ember_30m_hero/best.pt
               run $PY scripts/plot_loss.py experiments/runs/ember_30m_hero.csv; }
do_scaling() { for cfg in scale_5m scale_12m ablate_30m_lr6e4 ablate_60m_lr6e4; do
                 run $PY src/train.py --config configs/$cfg.py
               done
               run $PY scripts/plot_scaling.py; }
do_ablations() { for cfg in ablation_12m_base ablation_12m_learnedpos ablation_12m_postnorm \
                            ablation_12m_layernorm ablation_12m_gelu ablation_12m_deep; do
                   run $PY src/train.py --config configs/$cfg.py
                 done; }
do_bench()   { run $PY scripts/benchmark.py
               run $PY scripts/plot_benchmarks.py; }
do_lora()    { run $PY scripts/finetune_lora.py; }
do_judge()   { for judge in gemini-2.5-flash gemini-2.5-flash-lite; do
                 run $PY -u scripts/judge_eval.py \
                   --ckpt-a checkpoints/ember_30m_hero/best.pt \
                   --ckpt-b checkpoints/lora_30m_r8/merged.pt \
                   --n 100 --judge-model "$judge" \
                   --name "hero_vs_lora_${judge##*-}"
               done; }

case "$stage" in
  tests)     do_tests ;;
  data)      do_data ;;
  ladder)    do_ladder ;;
  hero)      do_hero ;;
  gallery)   do_gallery ;;
  scaling)   do_scaling ;;
  ablations) do_ablations ;;
  bench)     do_bench ;;
  lora)      do_lora ;;
  judge)     do_judge ;;
  all)       do_tests; do_data; do_ladder; do_hero; do_gallery
             do_scaling; do_ablations; do_bench; do_lora; do_judge ;;
  *) echo "unknown stage: $stage (see header for options)"; exit 1 ;;
esac
echo; echo "stage '$stage' complete"
