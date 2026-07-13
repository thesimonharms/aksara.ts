# HF Space (SDK: docker) container for the Javanese TrOCR fine-tune pipeline.
# Hardware: assign "T4 small" in Space Settings → Hardware.
# Secrets:   set HF_TOKEN (write-perm) and HF_USERNAME in Space Settings → Secrets.

FROM python:3.11-slim

# System deps: git for HF Hub push, git-lfs for large model/dataset files,
# libsndfile1/tesseract useful if you ever add audio OCR or scanned-text OCR steps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs libsndfile1 tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

WORKDIR /app

# Cache deps layer separately
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Pipeline scripts + gradio front-end
COPY finetune_trocr.py /app/finetune_trocr.py
COPY generate_trocr_dataset.py /app/generate_trocr_dataset.py
COPY app.py /app/app.py
COPY label_pdfs.py /app/label_pdfs.py

# Font + PDF drop folders. Upload your .ttf/.otf and .pdf files into the Space
# repo, placing TTF/OTF under fonts/ and scanned manuscripts under pdfs/.
# (These Space-local dirs are independent of the gitignored dirs in aksara.ts —
# the Space repo is its own thing on HF.)
RUN mkdir -p /app/fonts /app/pdfs /app/trocr_dataset /app/pdf_labeled

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    HF_HUB_DISABLE_PROGRESS_BARS=1

EXPOSE 7860
ENTRYPOINT ["python", "app.py"]