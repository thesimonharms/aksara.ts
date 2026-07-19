#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(Path(__file__).resolve().parents[2] / ".env")
api = HfApi()
job_id = "6a57b91f85d9643ce16d519b"
chunks: list[str] = []
t0 = time.time()
for line in api.fetch_job_logs(job_id=job_id):
    chunks.append(line if isinstance(line, str) else str(line))
    if time.time() - t0 > 90:
        break
s = "".join(chunks)
print("chars", len(s))
losses = re.findall(r"\{'loss': [^}]+\}", s)
print("n_losses", len(losses))
if losses:
    print("first", losses[0])
    print("last", losses[-1])
ev = re.findall(r"\{'eval_loss':[^}]+\}", s)
print("n_eval", len(ev))
if ev:
    print("last_eval", ev[-1])
for key in ("Running final", "Final CER", "final_cer", "[OK] Model"):
    print(key, s.find(key))
idx = s.rfind("Running final")
print("final_snip:", repr(s[idx : idx + 800]) if idx >= 0 else "NOT FOUND")
# epochs seen
eps = re.findall(r"'epoch': ([0-9.]+)", s)
if eps:
    print("max_epoch_logged", max(float(x) for x in eps))
