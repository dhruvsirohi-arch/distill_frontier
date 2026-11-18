# vla-v1

Frozen copy of the first VLA, matching git tag `vla-v1`.

| File | What | Source run |
|---|---|---|
| `vla-v1-m2.pt` | two bowls, two moving pads, language-conditioned (the model in the latency report and the studio) | `outputs/vla/runs/m2_early/best.pt`, 20k + 8k + 6k steps |
| `vla-v1-m1.pt` | one bowl, one moving pad | `outputs/vla/runs/m1/best.pt`, step 12k |

The `.pt` files are not in git (160 MB each); they exist only on the training machine. Verify a copy with `sha256sum -c SHA256SUMS`.
Each checkpoint contains the EMA weights, normalization statistics, cached instruction embeddings and the full `vla` config it was trained with, so it loads with the code at this tag and nothing else:

```bash
git checkout vla-v1
.venv/bin/python -m vla.studio   --ckpt outputs/vla/models/vla-v1/vla-v1-m2.pt
.venv/bin/python -m vla.evaluate --ckpt outputs/vla/models/vla-v1/vla-v1-m2.pt --n 100 --tag v1
```

## Architecture

39.7 M parameters. DINOv2-S vision encoder (last 2 blocks trained) on 2 cameras × 2 frames (now and 160 ms earlier), frozen CLIP ViT-B/32 text encoder, 4-layer fusion transformer, 4-layer flow-matching decoder predicting 16 ticks of joint targets at 25 Hz; 4 are executed before replanning.

## Measured (see `results_m2.json`, `results_m1.json`, `latency.json`)

- M2: 100/100 on held-out seeds; 148/150 with the instruction swapped; 30/30 unseen phrasings; 30/30 mid-carry pad switches.
- Latency on an RTX 4090: 10.3 ms p50, 11.2 ms p99 (loop kept hot); 6.7 ms compiled in fp32.

## Known limits

- Trained only on pads moving 1–3 cm/s over ±4 cm. Holds to roughly 9 cm/s in informal tests; 0 of 4 at 12 cm/s.
- After a failed placement at high speed it can go for the wrong bowl.
- A second task in a scene where a bowl already sits on a pad usually fails.
- Reads the cameras once every 160 ms and acts open-loop in between.
