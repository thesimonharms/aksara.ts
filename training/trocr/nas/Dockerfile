# Hands-off TrOCR fine-tune on Intel Arc (XPU) for Linux NAS + Docker.
# Host needs Intel GPU compute drivers + /dev/dri accessible.
FROM intel/pytorch:latest

USER root
WORKDIR /workspace

# Train deps (torch already provided by the base image).
RUN python -m pip install --no-cache-dir -U pip \
 && python -m pip install --no-cache-dir \
      "transformers>=4.46,<5.0" \
      "datasets>=2.20" \
      "accelerate>=0.34" \
      "evaluate>=0.4" \
      "jiwer>=3.0" \
      "sentencepiece>=0.2.0" \
      "python-dotenv>=1.0" \
      "pillow>=10.0" \
      "huggingface_hub[hf_xet]>=0.24"

# Repo training code is bind-mounted at runtime to /workspace/trocr
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/trocr \
    HF_HOME=/workspace/hf-cache \
    TRANSFORMERS_CACHE=/workspace/hf-cache \
    HUGGINGFACE_HUB_CACHE=/workspace/hf-cache

COPY entrypoint.sh detect_gpu.sh train_v2_handsoff.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/entrypoint.sh \
              /usr/local/bin/detect_gpu.sh \
              /usr/local/bin/train_v2_handsoff.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["train"]
