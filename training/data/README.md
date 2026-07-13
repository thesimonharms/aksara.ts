# Manuscript Background PDFs (`training/data/`)

Place public-domain or archival Javanese manuscript scan PDFs in this directory (e.g., from Perpustakaan Nasional or institutional repositories).

When running `--mode generate_from_corpus`:
```bash
python training/javanese_ocr.py --mode generate_from_corpus \
  --corpus training/javanese_aksara.txt \
  --background_pdf training/PDFA.pdf training/data/scan1.pdf \
  --data_dir ./ocr_corpus \
  --num_samples 20000
```
The data generator will sample parchment patches across all provided PDFs to broaden the synthetic training distribution and improve robustness to real manuscript degradation.
