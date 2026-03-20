#!/usr/bin/env python3
"""
int4_train_test.py - Pure INT4 training experiment.
All weights stored and updated as int4. Uses STE for gradients.
"""

import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"


class STEQuantizeInt4(torch.autograd.Function):
    """Straight-Through Estimator for int4 quantization.

    Forward: quantize to int4 range [-8, 7]
    Backward: pass gradients through unchanged
    """

    @staticmethod
    def forward(ctx, x, scale):
        # Quantize: scale -> round -> clamp -> dequantize
        q = torch.round(x / scale).clamp(-8, 7)
        return q * scale  # dequantized output (but represents int4 values)

    @staticmethod
    def backward(ctx, grad_output):
        # STE: pass gradient through as-is
        return grad_output, None


class Int4Linear(nn.Module):
    """Linear layer with int4 weights (STE training)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full-precision "shadow" weights for gradient flow
        self.weight_fp = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Initialize
        nn.init.kaiming_uniform_(self.weight_fp, a=5**0.5)

    def quantized_weight(self):
        """Get int4-quantized weight (with STE in backward)."""
        # Compute scale from current weight range
        w = self.weight_fp.data
        scale = w.abs().max() / 7.0  # map max to int4 range
        scale = scale.clamp(min=1e-8)
        # STE quantize
        return STEQuantizeInt4.apply(self.weight_fp, scale)

    def forward(self, x):
        w_q = self.quantized_weight()
        return nn.functional.linear(x, w_q, self.bias)

    def get_int4_storage(self):
        """Get actual int4 weights for saving."""
        w = self.weight_fp.data
        scale = w.abs().max() / 7.0
        scale = scale.clamp(min=1e-8)
        q = torch.round(w / scale).clamp(-8, 7).to(torch.int8)
        return q, scale


class Int4Embedding(nn.Module):
    """Embedding with int4 weights."""

    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight_fp = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight_fp, std=0.02)

    def quantized_weight(self):
        w = self.weight_fp.data
        scale = w.abs().max() / 7.0
        scale = scale.clamp(min=1e-8)
        return STEQuantizeInt4.apply(self.weight_fp, scale)

    def forward(self, x):
        w_q = self.quantized_weight()
        return nn.functional.embedding(x, w_q)


class Int4LSTMCell(nn.Module):
    """Single LSTM cell with int4 weights."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Combined gates: input, forget, cell, output
        self.w_ih = Int4Linear(input_size, 4 * hidden_size)
        self.w_hh = Int4Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, hx=None):
        if hx is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
            c = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        else:
            h, c = hx

        gates = self.w_ih(x) + self.w_hh(h)
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class Int4TinyLSTM(nn.Module):
    """Full int4 LSTM language model."""

    def __init__(self, vocab_size, d_model=32, num_layers=1):
        super().__init__()
        self.embedding = Int4Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            input_size = d_model if i == 0 else d_model
            self.layers.append(Int4LSTMCell(input_size, d_model))
        self.fc = Int4Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x, hidden=None):
        # x: (batch, seq)
        emb = self.embedding(x)  # (batch, seq, d_model)
        batch, seq, _ = emb.shape

        if hidden is None:
            hidden = [
                (
                    torch.zeros(batch, self.d_model, device=x.device),
                    torch.zeros(batch, self.d_model, device=x.device),
                )
                for _ in self.layers
            ]

        outputs = []
        for t in range(seq):
            h = emb[:, t, :]
            for l, layer in enumerate(self.layers):
                h_new, c_new = layer(h, hidden[l])
                hidden[l] = (h_new, c_new)
                h = h_new
            outputs.append(h)

        out = torch.stack(outputs, dim=1)  # (batch, seq, d_model)
        logits = self.fc(out)
        return logits, hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_int4(self):
        """Calculate model size if stored as int4."""
        total_int4 = 0
        total_scale = 0
        for module in self.modules():
            if isinstance(module, (Int4Linear, Int4Embedding)):
                n = module.weight_fp.numel()
                total_int4 += n * 4 / 8  # 4 bits per weight = 0.5 bytes
                total_scale += 4  # one float32 scale per layer
        # biases are float32
        for name, param in self.named_parameters():
            if "bias" in name:
                total_scale += param.numel() * 4
        return (total_int4 + total_scale) / 1024  # KB


class TextDataset(Dataset):
    def __init__(self, tokens, seq_length=64):
        self.tokens = tokens
        self.seq_length = seq_length

    def __len__(self):
        return max(1, len(self.tokens) - self.seq_length)

    def __getitem__(self, idx):
        x = self.tokens[idx : idx + self.seq_length]
        y = self.tokens[idx + 1 : idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


def main():
    print("=" * 60)
    print("INT4 Pure Training Experiment")
    print("=" * 60)

    # Load data
    info_file = DATA_DIR / "info.json"
    with open(info_file) as f:
        info = json.load(f)
    tokens = np.load(info["tokenized_file"])
    vocab_size = info["vocab_size"]
    print(f"Loaded {len(tokens)} tokens, vocab_size={vocab_size}")

    # Small subset for quick test
    tokens = tokens[:100000]
    split = int(len(tokens) * 0.9)
    train_tokens = tokens[:split]
    val_tokens = tokens[split:]

    seq_length = 64
    batch_size = 4
    d_model = 32
    max_steps = 200
    epochs = 2

    train_ds = TextDataset(train_tokens, seq_length)
    val_ds = TextDataset(val_tokens, seq_length)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")

    # Create int4 model
    model = Int4TinyLSTM(vocab_size, d_model=d_model, num_layers=1)
    num_params = model.count_parameters()
    int4_size_kb = model.get_model_size_int4()
    print(f"Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    print(f"INT4 model size: {int4_size_kb:.1f} KB")
    fp32_size_kb = num_params * 4 / 1024
    print(
        f"FP32 model size: {fp32_size_kb:.1f} KB (compression: {fp32_size_kb / int4_size_kb:.1f}x)"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    start = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        count = 0
        t0 = time.time()

        for i, (x, y) in enumerate(train_dl):
            if i >= max_steps:
                break
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            count += 1
            if count % 50 == 0:
                print(
                    f"  Epoch {epoch + 1} Step {count}: loss={total_loss / count:.4f}"
                )

        avg_train = total_loss / max(count, 1)

        # Eval
        model.eval()
        val_loss = 0
        val_count = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(val_dl):
                if i >= max_steps:
                    break
                logits, _ = model(x)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                val_loss += loss.item()
                val_count += 1

        avg_val = val_loss / max(val_count, 1)
        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch + 1}: train_loss={avg_train:.4f}, val_loss={avg_val:.4f}, time={elapsed:.1f}s"
        )

    total_time = time.time() - start

    # Test generation
    print("\n--- Generation Test ---")
    model.eval()
    prompt = "Once upon a time"
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokens_in = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens_in], dtype=torch.long)
    generated = list(tokens_in)

    with torch.no_grad():
        for _ in range(50):
            context = input_ids[:, -64:]
            logits, _ = model(context)
            next_logits = logits[0, -1, :] / 0.8
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(next_token)
            input_ids = torch.tensor([generated], dtype=torch.long)

    print(f"Prompt: {prompt}")
    print(f"Output: {tokenizer.decode(generated)}")

    # Summary
    print("\n" + "=" * 60)
    print(f"val_loss:         {avg_val:.4f}")
    print(f"training_seconds: {total_time:.1f}")
    print(f"num_params_M:     {num_params / 1e6:.2f}")
    print(f"int4_size_kb:     {int4_size_kb:.1f}")
    print(f"compression:      {fp32_size_kb / int4_size_kb:.1f}x vs float32")
    print("=" * 60)


if __name__ == "__main__":
    main()
