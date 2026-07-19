# NAS / Docker hands-off TrOCR v2 (Intel Arc XPU)

Fire-and-forget fine-tune on dad’s Linux NAS once SSH + Arc (B70 via OcuLink
and/or onboard Arc) are available. **Starts from Hub v1**, mixes original + 180k
+ NusaAksara data, trains up to 15 epochs with early stopping, and pushes the
**best** checkpoint to `thesimonharms/trocr-javanese-synthetic-v2` (overwrite).

You should not need to say “do another epoch.”

## Recipe (why this one)

| Choice | Value | Why |
|--------|--------|-----|
| Base | `…-synthetic` (**v1**) | v2 continue-FT twice *hurt* CER even on 180k-val |
| Data | 180k + original (`×1`) + Nusa (`×8`) | Replay scoreboard domain + new unique lines + real OCR |
| LR | `1e-5` | Continue-FT at `2e-5` walked off v1’s basin |
| Epochs | 15 max, early-stop patience 3 on `eval_loss` | Hands-off; stops when val loss stalls |
| Push | every epoch + final best → Hub **v2** | Survive reboots; one Hub id to watch |
| Device | auto-pick discrete XPU (max VRAM, non-iGPU) | NAS may expose iGPU + OcuLink B70 |

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
