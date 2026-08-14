# NAS / Docker hands-off TrOCR (Intel XPU)

Fire-and-forget fine-tune on the NAS.

| Cook | Hub model | Data | Status |
|------|-----------|------|--------|
| **v6** | [`trocr-javanese-synthetic-v6`](https://huggingface.co/thesimonharms/trocr-javanese-synthetic-v6) | `javanese-synthetic-exact` (≤12 aksara, 384×384) | **Current — 96.0% exact** |
| v5 | never published | `javanese-synthetic-hq` | Stopped (same failure mode) |
| v4–v1 | deleted from Hub | mixed / long wiki lines | Superseded |

v6 starts from `microsoft/trocr-small-printed` (not large). Capacity was not
the bottleneck; decoder start / anti-loop / unshifted CE / square canvases
were. Small trains fast enough for many epochs in one NAS run. Target:
**exact-match rate**, not CER. Findings and limitations are on the Hub card.

**Free-gen scoring:** `local_verify_large.py` / `score_epoch_checkpoints.py` attach
`generation_utils.NoRunawayMarksLogitsProcessor` at inference time (blocks cecak /
sandhangan runaway loops). Teacher-forced `eval_loss` mid-train is unchanged.
`no_repeat_ngram_size` stays `0` in saved `generation_config` (byte-BPE footgun).
Set `ANTI_LOOP=0` to reproduce unguarded generate().
## Recipe (v6 — current)

| Choice | Value | Why |
|--------|--------|-----|
| Base | `microsoft/trocr-small-printed` | Printed prior; ~9× smaller than large → more epochs |
| Data | `javanese-synthetic-exact` | ≤12 aksara, 384×384, no manuscript bg, aksara-only labels |
| Phase 0 | overfit 32, 400 ep, **discard** | Fail fast if tokenizer/generate/pad/loss is still broken |
| Phase A | 3 ep, **unfrozen**, LR `5e-5` | Frozen DeiT-small collapsed; must adapt encoder |
| Phase B | **12 ep**, unfrozen, LR `2e-5` | From A final; no tokenizer re-expand |
| Score | exact-match on exact-val | Gate is 90% exact, not 10% CER |
| Device | **NAS iGPU XPU only** | Do not train on the laptop |

## Recipe (v5 — reference, do not rerun)

| Choice | Value | Why |
|--------|--------|-----|
| Base | `microsoft/trocr-large-printed` | Printed prior matches synthetic lines |
| Data | `javanese-synthetic-hq` ×1 | Held-out text, length mix, degrade — no old mix drown |
| Phase A | 2 ep, freeze encoder, LR `3e-5` | New tokens need decoder learning first |
| Phase B | **2 ep**, unfreeze, LR `1e-5` | v4 ep3 free-gen was flat; stop earlier |
| Score | HQ validation (anti-loop) | In-domain gate for synthetic capability |
| Device | **NAS iGPU XPU only** | Do not train on the laptop |

## Recipe (v4 — reference)

| Choice | Value | Why |
|--------|--------|-----|
| Base | `microsoft/trocr-large-printed` | Printed prior matches synthetic lines; large capacity |
| Data | original ×6 + Nusa ×8 + 180k ×1 | Scoreboard domain upweighted |
| Phase A | 2 ep, freeze encoder, LR `3e-5` | New tokens need decoder learning first (~42h/ep @ batch 8) |
| Phase B | 3 ep, unfreeze, LR `1e-5` | Rewire vision to Aksara (wall-clock fit) |
| Push | every epoch + `epoch-N` tags → Hub **v4** | Scoreboard escape hatch |
| Device | **NAS iGPU XPU only** (`/dev/dri`) | Do not train on the laptop |
| Mid-train gate | Laptop DirectML sniff only (`sniff_hub_epoch.ps1`) | Optional free-gen check; never steals NAS XPU |

Expected: Linux XPU + B70 should be **much** faster than Windows eGPU B60
(~1.7 it/s eager/fp32). First bring-up may still need `TROCR_ATTN=eager` /
`TROCR_FORCE_FP32=1` if SDPA/bf16 misbehave — env knobs are wired.

## Host prerequisites (NAS Linux)

1. Intel GPU compute stack for Arc / Battlemage (kernel `xe` or `i915` as required).
2. Docker Engine + Compose plugin.
3. Devices visible: `ls /dev/dri/renderD*`
4. Repo checked out (this folder present).
5. `HF_TOKEN` with write access to your model namespace.

## One-time setup

```bash
ssh user@nas
cd /path/to/aksara.ts/training/trocr/nas

# Fix CRLF if scripts were edited on Windows:
sed -i 's/\r$//' entrypoint.sh detect_gpu.sh train_v2_handsoff.sh

export HF_TOKEN=hf_xxx
export RENDER_GID="$(stat -c %g /dev/dri/renderD128)"   # or the OcuLink render node

docker compose build
docker compose run --rm trocr-train detect    # must show torch.xpu + Arc name
docker compose run --rm trocr-train smoke     # matmul on XPU
```

Prefer the **B70** render node if both cards appear: check `detect` output and
set `ONEAPI_DEVICE_SELECTOR=level_zero:N` in `docker-compose.yml` if the
auto-pick is wrong.

## Hands-off train (forget until Hub updates)

```bash
cd /path/to/aksara.ts/training/trocr/nas
export HF_TOKEN=hf_xxx
export RENDER_GID="$(stat -c %g /dev/dri/renderD128)"

# v6 (short clean exact-match) — preferred
docker compose up -d trocr-train-v6
docker compose logs -f trocr-train-v6

# v4 (legacy mix) — only if intentionally re-running
# docker compose up -d trocr-train
```

- Detach anytime; container keeps training.
- Fresh `trocr_v5_stage_{a,b}` dirs — no accidental Phase A resume from v4.
- `restart: "no"` — finished/score exit does not re-enter train.
- Score-only: `docker compose run --rm trocr-train-v6` with `SCORE_ONLY=1`
- Done when logs show `[train] DONE v6` and Hub `…-synthetic-v6` has `evals/scores.csv`.

Watch Hub: https://huggingface.co/thesimonharms/trocr-javanese-synthetic-v6

## Verify after it finishes (on NAS or laptop)

```bash
# v6 in-domain (synthetic-exact val)
HUB_MODEL_ID=thesimonharms/trocr-javanese-synthetic-v6 \
DATASET_NAME=thesimonharms/javanese-synthetic-exact \
N_SAMPLES=1500 python local_verify_large.py
```

Gate (v6): exact-match on exact-val should stay ≥ **90%** (the published B-final is **96.0%**). If a recook misses that, change data/geometry/loss — do not switch to `trocr-large`.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `torch.xpu` false in container | Pass `/dev/dri`, correct `RENDER_GID`, host drivers |
| Picks iGPU instead of B70 | `ONEAPI_DEVICE_SELECTOR=level_zero:N` from `detect` |
| SDPA / bf16 crash | `TROCR_ATTN=eager TROCR_FORCE_FP32=1` |
| OOM / DEVICE_LOST | Lower `TROCR_BATCH_SIZE` (16→8), enable GC in script |
| Empty Hub push | Ensure `HF_TOKEN` write scope; check logs volume |

## Layout

```
nas/
  Dockerfile              # intel/pytorch + HF train deps
  docker-compose.yml      # DRI passthrough + volumes
  entrypoint.sh           # detect | train | smoke
  detect_gpu.sh           # list XPUs, prefer discrete
  train_v2_handsoff.sh
  train_v3_handsoff.sh
  train_v4_handsoff.sh
  train_v5_handsoff.sh    # HQ-only cook (stopped)
  train_v6_handsoff.sh    # exact-match cook (current)
  README.md               # this file
```
