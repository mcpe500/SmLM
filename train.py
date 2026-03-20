#!/usr/bin/env python3
"""
train.py - CPU-friendly training loop for small language model.
Constrained to <10M params and <3.5GB RAM.
Baseline: LSTM-based model with ~1-2M params.
"""

import os
import sys
import time
import json
import resource
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Force single-threaded for CPU constraint
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


class TextDataset(Dataset):
    """Simple dataset that yields sequences of tokens."""

    def __init__(self, tokens, seq_length=128):
        self.tokens = tokens
        self.seq_length = seq_length

    def __len__(self):
        return max(1, len(self.tokens) - self.seq_length)

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_length]
        y = self.tokens[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class TinyLSTM(nn.Module):
    """Minimal LSTM language model."""

    def __init__(self, vocab_size, d_model=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.lstm = nn.LSTM(
            d_model,
            d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TinyTransformer(nn.Module):
    """Minimal Transformer language model."""

    def __init__(
        self,
        vocab_size,
        d_model=64,
        nhead=2,
        num_layers=2,
        dim_feedforward=128,
        max_len=512,
        dropout=0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.max_len = max_len

    def forward(self, x, hidden=None):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)

        emb = self.embedding(x) + self.pos_encoding(positions)

        # Causal mask
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()

        out = self.transformer(emb, mask=mask)
        logits = self.fc(out)
        return logits, None

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_peak_ram_mb():
    """Get peak RAM usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss is in KB on Linux
    return usage.ru_maxrss / 1024


def load_data():
    """Load tokenized data."""
    info_file = DATA_DIR / "info.json"
    if not info_file.exists():
        print("ERROR: Run prepare.py first!")
        sys.exit(1)

    with open(info_file) as f:
        info = json.load(f)

    tokens = np.load(info["tokenized_file"])
    vocab_size = info["vocab_size"]

    print(f"Loaded {len(tokens)} tokens, vocab_size={vocab_size}")
    return tokens, vocab_size


def train_epoch(
    model, dataloader, optimizer, criterion, device, grad_accum_steps=4, max_steps=None
):
    """Train one epoch with gradient accumulation."""
    model.train()
    total_loss = 0
    num_batches = 0
    optimizer.zero_grad()

    for i, (x, y) in enumerate(dataloader):
        if max_steps is not None and num_batches >= max_steps:
            break

        x, y = x.to(device), y.to(device)

        logits, _ = model(x)
        loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
        loss = loss / grad_accum_steps
        loss.backward()

        if (i + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

        if num_batches % 100 == 0:
            print(
                f"  Batch {num_batches}, loss={total_loss / num_batches:.4f}, RAM={get_peak_ram_mb():.0f}MB"
            )

    return total_loss / max(num_batches, 1)


def evaluate(model, dataloader, criterion, device, max_steps=None):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if max_steps is not None and num_batches >= max_steps:
                break
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--max-steps", type=int, default=5000, help="Max training steps per epoch"
    )
    args = parser.parse_args()

    start_time = time.time()
    device = torch.device("cpu")

    print("=" * 60)
    print("SLM Training - CPU Constrained")
    print("=" * 60)
    print(f"Architecture: {args.arch}")
    print(f"d_model: {args.d_model}, layers: {args.num_layers}")
    print(f"seq_length: {args.seq_length}, batch_size: {args.batch_size}")
    print(f"grad_accum: {args.grad_accum}, epochs: {args.epochs}")
    print(f"lr: {args.lr}")
    print(f"Initial RAM: {get_peak_ram_mb():.0f} MB")

    # Load data
    tokens, vocab_size = load_data()

    # Split train/val (90/10)
    split = int(len(tokens) * 0.9)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    print(f"Train tokens: {len(train_tokens)}, Val tokens: {len(val_tokens)}")

    # Create datasets
    train_dataset = TextDataset(train_tokens, seq_length=args.seq_length)
    val_dataset = TextDataset(val_tokens, seq_length=args.seq_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Create model
    if args.arch == "lstm":
        model = TinyLSTM(vocab_size, d_model=args.d_model, num_layers=args.num_layers)
    else:
        model = TinyTransformer(
            vocab_size,
            d_model=args.d_model,
            nhead=2,
            num_layers=args.num_layers,
            dim_feedforward=args.d_model * 2,
        )

    num_params = model.count_parameters()
    num_params_M = num_params / 1e6

    print(f"\nModel parameters: {num_params:,} ({num_params_M:.2f}M)")

    if num_params_M > 10:
        print("WARNING: Model exceeds 10M parameter limit!")

    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    # Training loop
    for epoch in range(args.epochs):
        epoch_start = time.time()
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            grad_accum_steps=args.grad_accum,
            max_steps=args.max_steps,
        )

        # Evaluate
        val_loss = evaluate(
            model, val_loader, criterion, device, max_steps=args.max_steps
        )

        scheduler.step()
        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time
        current_ram = get_peak_ram_mb()

        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Epoch time: {epoch_time:.1f}s, Total: {elapsed:.1f}s")
        print(f"  Peak RAM:   {current_ram:.1f} MB")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")

        # Safety check
        if current_ram > 3500:
            print("WARNING: RAM exceeds 3.5GB limit!")
            break

        # Timeout check (60 min)
        if elapsed > 3600:
            print("Timeout reached (60 min)")
            break

    total_time = time.time() - start_time
    peak_ram = get_peak_ram_mb()

    # Output summary
    print("\n" + "=" * 60)
    print(f"val_loss:         {best_val_loss:.4f}")
    print(f"training_seconds: {total_time:.1f}")
    print(f"peak_ram_mb:      {peak_ram:.1f}")
    print(f"num_params_M:     {num_params_M:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
