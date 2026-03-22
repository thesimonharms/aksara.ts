"""
Loads the saved model checkpoint and exports to ONNX.
Run this separately after training completes.

Usage:
    python training/export.py data/jv.txt
"""

import json
import sys
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from pathlib import Path

EMBED_DIM  = 64
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT    = 0.3


class SegmenterModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm       = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.drop       = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths=None):
        emb = self.drop(self.embedding(x))
        if lengths is not None:
            packed  = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _  = self.lstm(packed)
            out, _  = pad_packed_sequence(out, batch_first=True)
        else:
            out, _  = self.lstm(emb)
        return self.classifier(self.drop(out)).squeeze(-1)


data_path = sys.argv[1] if len(sys.argv) > 1 else "data/jv.txt"
print(f"Rebuilding vocab from: {data_path}")
lines     = Path(data_path).read_text("utf-8").splitlines()
char_set  = set()
for line in lines:
    char_set.update(c for c in line if c not in (" ", "\r"))
vocab     = ["<PAD>", "<UNK>"] + sorted(char_set)
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

Path("model").mkdir(exist_ok=True)
Path("model/vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
print("Saved: model/vocab.json")

model = SegmenterModel(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model.load_state_dict(torch.load("model/segmenter.pt", map_location="cpu"))
model.eval()

dummy = torch.zeros(1, 20, dtype=torch.long)

print("Exporting...")
torch.onnx.export(
    model,
    dummy,
    "model/segmenter.onnx",
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={
        "input":  {0: "batch", 1: "seq_len"},
        "logits": {0: "batch", 1: "seq_len"},
    },
    opset_version=14,
    dynamo=False,
)

print("Saved: model/segmenter.onnx")
