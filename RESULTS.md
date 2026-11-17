# Results (run of 2026-11-16/17, Apple M5 24 GB, MPS)

## Headline

| held-out seeds 100000–100099 (greedy, strict JSON) | Qwen3-1.7B base (before) | Qwen3-1.7B + LoRA (after) | scripted expert |
|---|---|---|---|
| **success / 100** | **0** | **90** | **100** |
| mean steps to success | – | 10.02 | 10.52 |
| fail: failed_grasp | 96 | 3 | 0 |
| fail: dropped_bowl | 0 | 5 | 0 |
| fail: missed_target | 0 | 2 | 0 |
| fail: invalid_json | 4 | 0 | 0 |
| fail: step_limit | 0 | 0 | 0 |

**Baseline.** The untuned base model gets the same system prompt, the same seeds and the same greedy decoding
(`python -m eval.evaluate --base` → `outputs/results_base.json`). It scores 0/100. In 66 of 100 scenes it moves
the empty gripper to the target coordinates, opens and says `done` without ever touching the bowl. It also
produced 13 invalid actions in 4 episodes.

After fine-tuning, Qwen produced **0 invalid JSON actions** over all 100 episodes. Per-cell success (5×5 spawn grid) is in
`outputs/results.json`; Qwen's failures are scattered across the workspace, with no single cell
failing on every seed.

## Latency (`python -m eval.latency` → `outputs/latency.json`, every token in `outputs/latency_decisions.jsonl`)

Setup: one robot (batch size 1), greedy decoding, merged Qwen3-1.7B + LoRA in bf16 on the Apple M5 GPU (MPS).
Thinking is off, so there are no reasoning tokens and every output token is part of the action JSON. The run
covers all 1,041 decisions of the 100 held-out episodes. Decoding runs token by token with a device sync after
each token; the first 40 decisions were checked and are 40/40 identical to `model.generate`. Success at batch 1
is also 90/100.

| | mean | p50 | p90 | p99 |
|---|---|---|---|---|
| **per decision (observation → complete action)** | 779 ms | **889 ms** | 1167 ms | 1427 ms |
| time to first token (prefill of ~234 prompt tokens) | 148 ms | 143 ms | 192 ms | 275 ms |
| per output token after the first | 38.4 ms | 36.5 ms | 47.3 ms | 55.1 ms |

| action | decisions | output tokens | p50 latency |
|---|---|---|---|
| move_to | 625 | 24 | 993 ms |
| open_gripper | 208 | 8 | 416 ms |
| close_gripper | 116 | 8 | 389 ms |
| done | 92 | 6 | 323 ms |

* Model time per episode: p50 8.0 s, against 6.3 s of simulated arm motion. The model currently thinks while
  the arm is stopped.
* Latency is set by the number of output tokens. In a `move_to`, 14 of the 25 tokens are fixed JSON syntax, and
  Qwen writes coordinates one digit per token. A compact action format or constrained decoding that fills in the
  fixed syntax would bring a `move_to` to about 11 generated tokens, roughly 0.5 s.
* The numbers drifted up during the run: a 3-episode smoke test just before it measured 106 ms to first token
  and 32.7 ms per token, probably because of thermal load or background load on the laptop. The table reports
  the slower full run.

## Pipeline numbers

| stage | result |
|---|---|
| expert check (seeds 0–49) | 49 / 50 = **98 %** (1 failed_grasp) |
| raw generation (seeds 0–1999, 9 workers) | 2000 episodes, 1989 successful (99.45 %), 111 s |
| quality filters | 1969 / 2000 pass (drops: low_place_margin 12, failed_grasp 10, collision 8, step_limit 1) |
| curated selection | **700 episodes** (100 bins × 7, 0 near-duplicates, no top-up needed) → 630 train / 70 val |
| training pairs | **6,578 train** / 733 val (state → action) |
| SFT | Qwen/Qwen3-1.7B, LoRA r=16, 1,498 steps = **0.91 epochs**, bs 4, final val loss 0.237 |
| SFT wall-clock | **158.4 min** (budget 50 min — see below) |
| eval wall-clock | 23 min (Qwen 100 episodes in lockstep batches of 25 ≈ 15 min, expert ≈ 8 min incl. recording) |

Checkpoint success on 5 separate seeds (100500–100504) during training: 0/5 → 1/5 → 4/5 → 5/5 → 4/5
at steps 0 / 374 / 749 / 1124 / 1498. Videos: `outputs/videos/train_ckpt_<step>/`.

## Deviations from the spec (stated plainly)

* **Epochs.** The step-20 probe measured 1.83 s/step → 2 epochs ≈ 105 min. The budget guard cut training
  to 1,498 steps (0.91 epochs) instead of the requested 2–3.
* **Time budget exceeded anyway.** Mid-run the machine ran out of RAM (browsers held ~19 GB of swap), the
  trainer was paged out and steps slowed from ~2 s to ~48 s, giving 158 min total. The guard only
  estimated once, at step 20. `train/sft.py` now also enforces a hard wall-clock stop, stops gracefully on
  SIGINT/SIGTERM, and saves the adapter at every checkpoint.
* **Model name.** "Qwen3-1.7B-Instruct" does not exist; `Qwen/Qwen3-1.7B` with thinking disabled was used.
* **Arm on a 12 cm pedestal** (wrist-flex limit; see README).

## Failure analysis (Qwen)

* All 10 failures are physical (no invalid JSON). 8 of the 10 run to the 15-action cap; they are
  labelled by their cause (bowl dropped / never grasped) rather than `step_limit`.
* The checkpoint videos show the typical pattern. After a missed or slipped grasp the model keeps
  executing the "carry to table" sequence even though the observation says `grasped:false`. The expert
  succeeds ~99 % of the time, so recovery states are almost absent from the demos (behaviour-cloning
  distribution shift).
* Likely fixes, in order of cost: DAgger (roll out Qwen, label the states it visits with the expert,
  retrain); inject perturbations or forced grasp failures into expert demos; train the full 2 epochs
  (needs ~105 min on this machine, or switch to Qwen3-0.6B).

## Artifacts (local; large files are not committed)

* Curation report: [`outputs/curation_report.md`](outputs/curation_report.md) (+ plots in `outputs/curation/`)
* Eval gallery: `outputs/eval_gallery.html` (100 + 100 low-res clips in `outputs/videos/eval/`, filter by policy/outcome/failure)
* `outputs/results.json` (committed), `outputs/train_summary.json` (committed), `outputs/eval_episodes.jsonl`
* LoRA adapter `outputs/qwen_lora/adapter/`, merged model `outputs/qwen_lora/merged/`
