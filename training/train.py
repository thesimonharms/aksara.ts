"""
Javanese word segmenter — BiLSTM character model.

Trains on Latin-script Javanese text (with spaces) and learns to predict
where word boundaries belong in unsegmented text. Exports to ONNX so the
TypeScript library can run inference without a Python dependency.

Usage:
    python training/train.py data/jv.txt

Output:
    model/segmenter.onnx   — inference model for TypeScript (onnxruntime-node)
    model/vocab.json       — character vocabulary
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

# ─── Device: AMD GPU via DirectML, CUDA, or CPU ──────────────────────────────

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
else:
    device = torch.device("cpu")
    print("Device: CPU")

# ─── Hyperparameters ─────────────────────────────────────────────────────────

EMBED_DIM   = 64
HIDDEN_DIM  = 128
NUM_LAYERS  = 2
DROPOUT     = 0.3
EPOCHS      = 30
BATCH_SIZE  = 64
LR          = 1e-3
THRESHOLD   = 0.5

# ─── Data ────────────────────────────────────────────────────────────────────

def load_data(path: str):
    lines = Path(path).read_text("utf-8").splitlines()
    lines = [l for l in lines if l.strip()]

    # Build character vocabulary
    char_set = set()
    for line in lines:
        char_set.update(c for c in line if c not in (" ", "\r"))
    vocab    = ["<PAD>", "<UNK>"] + sorted(char_set)
    char2idx = {c: i for i, c in enumerate(vocab)}

    sequences, labels_list = [], []
    for line in lines:
        chars, labels = [], []
        for ch in line.rstrip("\r"):
            if ch == " ":
                if labels:
                    labels[-1] = 1          # space follows this character
            else:
                chars.append(char2idx.get(ch, 1))
                labels.append(0)
        if chars:
            sequences.append(torch.tensor(chars,  dtype=torch.long))
            labels_list.append(torch.tensor(labels, dtype=torch.float))

    all_labels  = torch.cat(labels_list)
    space_rate  = all_labels.mean().item()
    print(f"Corpus:     {len(lines)} lines, {sum(len(s) for s in sequences):,} positions")
    print(f"Vocabulary: {len(vocab)} characters")
    print(f"Space density: {space_rate * 100:.1f}%")

    return sequences, labels_list, vocab, char2idx, space_rate


class SegmentDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs   = seqs
        self.labels = labels

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


def collate(batch):
    seqs, lbls = zip(*batch)
    lengths     = torch.tensor([len(s) for s in seqs])
    seqs_pad    = pad_sequence(seqs, batch_first=True, padding_value=0)
    lbls_pad    = pad_sequence(lbls, batch_first=True, padding_value=-1)
    return seqs_pad, lbls_pad, lengths

# ─── Model ───────────────────────────────────────────────────────────────────

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
            packed      = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _      = self.lstm(packed)
            out, _      = pad_packed_sequence(out, batch_first=True)
        else:
            out, _      = self.lstm(emb)
        return self.classifier(self.drop(out)).squeeze(-1)

# ─── Training ────────────────────────────────────────────────────────────────

data_path = sys.argv[1] if len(sys.argv) > 1 else "data/jv.txt"
print(f"\nLoading: {data_path}\n")
sequences, labels_list, vocab, char2idx, space_rate = load_data(data_path)

# 85/15 train/val split
n_val       = int(len(sequences) * 0.15)
train_ds    = SegmentDataset(sequences[n_val:], labels_list[n_val:])
val_ds      = SegmentDataset(sequences[:n_val], labels_list[:n_val])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

model     = SegmenterModel(len(vocab), EMBED_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

pos_weight = torch.tensor((1 - space_rate) / space_rate).to(device)
criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

n_params = sum(p.numel() for p in model.parameters())
print(f"\nParameters: {n_params:,}")
print(f"pos_weight: {pos_weight.item():.2f}x\n")
print("Training...\n")

best_val_acc = 0.0
Path("model").mkdir(exist_ok=True)

for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    total_loss = 0.0
    for seqs, lbls, lengths in train_loader:
        seqs, lbls = seqs.to(device), lbls.to(device)
        optimizer.zero_grad()
        logits  = model(seqs, lengths)
        mask    = lbls != -1
        loss    = criterion(logits[mask], lbls[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    # ── Validate ──
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for seqs, lbls, lengths in val_loader:
            seqs, lbls = seqs.to(device), lbls.to(device)
            logits  = model(seqs, lengths)
            mask    = lbls != -1
            preds   = (torch.sigmoid(logits[mask]) > THRESHOLD).float()
            correct += (preds == lbls[mask]).sum().item()
            total   += mask.sum().item()

    val_acc  = correct / total
    avg_loss = total_loss / len(train_loader)
    scheduler.step(avg_loss)
    marker = " ✓" if val_acc > best_val_acc else ""
    print(f"  Epoch {epoch + 1:2d}  loss: {avg_loss:.4f}  val_acc: {val_acc:.4f}{marker}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model/segmenter.pt")

print(f"\nBest val_acc: {best_val_acc:.4f}")
print("Saved: model/segmenter.pt")
Path("model/vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
print("Saved: model/vocab.json")

# ─── ONNX export ─────────────────────────────────────────────────────────────
# Load best checkpoint, move to CPU (ONNX export doesn't need a device),
# and export with dynamic batch + sequence length axes so TypeScript inference
# can call it on sequences of any length.

print("\nExporting to ONNX...")
model.load_state_dict(torch.load("model/segmenter.pt", map_location="cpu"))
model.eval().cpu()

dummy = torch.zeros(1, 20, dtype=torch.long)
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
    opset_version=17,
)

print("Saved: model/segmenter.onnx")

# ─── Demo ────────────────────────────────────────────────────────────────────

def segment(text: str, threshold: float = THRESHOLD) -> str:
    chars   = list(text.replace(" ", ""))
    indices = torch.tensor([[char2idx.get(c, 1) for c in chars]], dtype=torch.long)
    with torch.no_grad():
        logits = model(indices)
    probs  = torch.sigmoid(logits[0]).tolist()
    result = ""
    for i, ch in enumerate(chars):
        result += ch
        if probs[i] > threshold and i < len(chars) - 1:
            result += " "
    return result


demos = [
    "balèkakédatakanggo",
    "kasalahankakèhankategori",
    "oranakacasingpadhakaro",
    "tambahlebonsaransindikasisakakontènpinilihwiki",
    "botngowahténmplatténmplat",
]

print("\nDemo segmentation:\n")
for s in demos:
    print(f"  input : {s}")
    print(f"  output: {segment(s)}")
    print()
