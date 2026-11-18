# Results

## VLA: bowl onto a moving pad (branch `vla-mujoco`)

A 39.7 M-parameter vision-language-action model, trained from scripted-expert demonstrations in
MuJoCo, places the named bowl on the named moving pad in 100 of 100 held-out episodes at about
10 ms per inference on an RTX 4090.

Input is two 224×224 RGB cameras (scene and wrist), six joint positions and a text instruction.
Output is six joint position targets at 25 Hz. The model sees no privileged state at run time.

### Success on 100 held-out seeds

| Milestone | Scene | VLA | Scripted expert |
|---|---|---|---|
| M1 | one bowl, one moving pad | 100/100 | 100/100 |
| M2 | two bowls, two moving pads, instruction names which goes where | 100/100 | 100/100 |

Success means the named bowl is resting on the named pad, upright, released, centred within 3 cm
and moving with the pad, held for 2 s. Pads move at 1–3 cm/s; success is 100% in every speed band.
Episodes take 5.2 s on average, the same as the expert.

### Language (M2)

| Test | Result |
|---|---|
| Same scene, instruction changed to each of the other three bowl/pad combinations (50 scenes) | 148/150; both failures picked the wrong bowl |
| Three phrasings never seen in training | 30/30 |
| Instruction changed to the other pad 10 ticks after the bowl is grasped (never seen in training) | 30/30 |

The last row is the interface for a reasoning layer: `VLAPolicy.set_instruction()` can be called
at any tick and forces an immediate replan (`vla/switch_demo.py`).

### Latency (RTX 4090, batch 1, bf16, GPU otherwise idle)

| Model | p50 | p90 | p99 | max |
|---|---|---|---|---|
| M1 | 9.7 ms | 10.3 ms | 18.4 ms | 39.9 ms |
| M2 | 10.4 ms | 10.8 ms | 11.2 ms | 34.6 ms |

Targets were p50 < 15 ms and p99 < 25 ms; the control tick is 40 ms. One inference yields 16 ticks
of actions, of which 4 are executed before replanning. `torch.compile` was not used.

### Ablations (M1 checkpoint)

| Setting | Success | p50 latency |
|---|---|---|
| Default: 4 flow steps, replan every 4 ticks | 100/100 | 9.7 ms |
| 2 flow steps | 100/100 | 7.4 ms |
| 1 flow step | 100/100 | 6.2 ms |
| Replan every 8 ticks | 99/100 | 12.7 ms |
| Replan every 16 ticks (whole chunk open loop) | 98/100 | 13.2 ms |

Success is over 100 seeds; latency for the non-default rows was re-measured over 30 seeds. Longer
replan intervals cost more per inference because the earlier frame is no longer in the token cache.
The no-history-frame ablation in the plan was not run.

### What made M2 work

The first M2 run (20,000 steps, same recipe as M1) reached only 50%: it always used the correct
pad but picked the bowl by position, ignoring the colour in the instruction. The cause: the
instruction decides the action only in the first few ticks of an episode (about 4% of samples).
After that the arm's own motion shows which bowl is the target, so the model could fit almost all
of the data without reading the bowl colour. Two changes, in `configs/vla_m2.yaml`:

1. A training-only grounding head that predicts the xy of the named bowl and pad. On its own this
   did not help (still 50% after 8,000 more steps), but it exposed the problem: at tick 0 its
   estimate sat at the midpoint between the bowls, at tick 10 it was within 5 mm.
2. Sampling the first 6 ticks of each episode 6× more often. With this, grounding appeared between
   3,000 and 4,500 further steps and closed-loop success went from 55% to 100%.

The benchmarked M2 checkpoint is therefore the result of 20,000 + 8,000 + 6,000 steps across three
runs (`m2` → `m2_aux` → `m2_early`). A single from-scratch run with the final config has not been done.

### Other caveats

- M1 was benchmarked at step 12,000 of 20,000 because a reboot interrupted training.
- M1's instruction carries no information, so its "unseen phrasing" result says nothing about language.
- Everything is in simulation with mild visual randomization (lighting, camera jitter, floor tint).
  Bowl and pad colours, pad positions and the pad's axis of motion are fixed.
- 87 of 3,000 M2 expert episodes failed under injected noise and were dropped from the training set.

### Reproduce

```bash
uv sync --frozen
C=configs/vla_m2.yaml
.venv/bin/python -m vla.generate --config $C --n 100 --split val
.venv/bin/python -m vla.generate --config $C --n 3000 --split train
.venv/bin/python -m vla.train --config $C --run m2
.venv/bin/python -m vla.evaluate --ckpt outputs/vla/runs/m2/best.pt --n 100 --video 8 --tag m2
.venv/bin/python -m vla.switch_demo --ckpt outputs/vla/runs/m2/best.pt
```

Drop `--config` for M1. Numbers are in `outputs/vla/results_*.json`, videos in `outputs/vla/videos/`.

## Qwen LoRA macro-action policy (original pipeline)

Text-only state in, about ten JSON macro-actions per episode out, static target: 92/100 against the
expert's 100/100. See `outputs/results.json`.
