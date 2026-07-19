#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

job_id = sys.argv[1] if len(sys.argv) > 1 else "6a57b91f85d9643ce16d519b"
api = HfApi()
j = api.inspect_job(job_id=job_id)
print("stage", getattr(j.status, "stage", j.status))

import time

chunks: list[str] = []
t0 = time.time()
for line in api.fetch_job_logs(job_id=job_id):
    chunks.append(line if isinstance(line, str) else str(line))
    if len(chunks) > 20000 or (time.time() - t0) > 45:
        break
text = "".join(chunks)
print("log_chars", len(text), "elapsed_fetch", round(time.time() - t0, 1))


losses = re.findall(r"\{'loss':[^}]+\}", text)
print("loss_count", len(losses))
if losses:
    print("latest_loss", losses[-1])

matches = list(re.finditer(r"(\d+)/41680[^\n]*?([\d.]+(?:it/s|s/it))", text))
if matches:
    print("latest_progress", matches[-1].group(0)[:160])

for bad in ("Traceback", "CUDA out of memory", "[ERROR]", "OOM"):
    if bad in text:
        print("FOUND", bad)

print("---TAIL---")
print(text[-2500:])
