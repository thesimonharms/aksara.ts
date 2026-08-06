# NAS / Docker hands-off TrOCR v4 (Intel XPU)

Fire-and-forget fine-tune on the NAS. **v4** starts from
`microsoft/trocr-large-printed` (printed prior for synthetic Aksara), expands the
Javanese tokenizer, freezes the encoder briefly, then unfreezes — Hub id
`thesimonharms/trocr-javanese-synthetic-v4`.

Mid-train free-gen gates: on the laptop, `.\sniff_hub_epoch.ps1 -Revision epoch-N`
(DirectML). Do not steal the NAS XPU for long verifies while training.

**Free-gen scoring:** `local_verify_large.py` / `score_epoch_checkpoints.py` attach
`generation_utils.NoRunawayMarksLogitsProcessor` at inference time (blocks cecak /
sandhangan runaway loops). Teacher-forced `eval_loss` mid-train is unchanged.
`no_repeat_ngram_size` stays `0` in saved `generation_config` (byte-BPE footgun).
Set `ANTI_LOOP=0` to reproduce unguarded generate().

## Recipe (v4)

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

# Optional overrides:
# export EPOCHS=20 LR=5e-6 TROCR_BATCH_SIZE=24

docker compose up -d trocr-train
docker compose logs -f trocr-train
```

- Detach anytime; container keeps training.
- If the box reboots, `docker compose up -d` again — script **resumes** from
  `checkpoint-*` under the `trocr_output` volume.
- Done when logs show `[train] DONE` and Hub `…-synthetic-v2` has a fresh
  “End of training” commit.

Watch Hub: https://huggingface.co/thesimonharms/trocr-javanese-synthetic-v2

## Verify after it finishes (on NAS or laptop)

```bash
# In-domain + original scoreboard
HUB_MODEL_ID=thesimonharms/trocr-javanese-synthetic-v2 \
DATASET_NAME=thesimonharms/javanese-dataset \
N_SAMPLES=1500 python local_verify_large.py
```

Gate: original-val CER should beat **v1 ~0.63** (and 180k-val beat **v1 ~0.61**).
If not, do **not** keep training the same recipe — change data/LR, don’t burn epochs.

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
  train_v2_handsoff.sh    # single command train → Hub v2
  README.md               # this file
```
