#!/usr/bin/env python3
"""
int2_train_test.py - Pure INT2 training experiment.
Only 4 quantization levels: -1.5, -0.5, 0.5, 1.5 (scaled by learnable factor)
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


class STEQuantizeInt2(torch.autograd.Function):
    """STE for int2: 4 levels centered at quantiles of normal dist.
    Levels: -1.5, -0.5, 0.5, 1.5 (then scaled)
    """

    @staticmethod
    def forward(ctx, x, scale):
        q = torch.round(x / scale * 2) / 2  # round to half-integers
        return q.clamp(-1.5, 1.5) * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class Int2Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_fp = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight_fp, a=5**0.5)

    def quantized_weight(self):
        scale = self.weight_fp.data.abs().mean() * 2
        scale = scale.clamp(min=1e-8)
        return STEQuantizeInt2.apply(self.weight_fp, scale)

    def forward(self, x):
        return nn.functional.linear(x, self.quantized_weight(), self.bias)


class Int2Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight_fp = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.normal_(self.weight_fp, std=0.02)

    def quantized_weight(self):
        scale = self.weight_fp.data.abs().mean() * 2
        scale = scale.clamp(min=1e-8)
        return STEQuantizeInt2.apply(self.weight_fp, scale)

    def forward(self, x):
        return nn.functional.embedding(x, self.quantized_weight())


class Int2LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.w_ih = Int2Linear(input_size, 4 * hidden_size)
        self.w_hh = Int2Linear(hidden_size, 4 * hidden_size)

    def forward(self, x, hx=None):
        if hx is None:
            h = torch.zeros(x.size(0), self.hidden_size, device=x.device)
            c = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        else:
            h, c = hx
        gates = self.w_ih(x) + self.w_hh(h)
        i, f, g, o = gates.chunk(4, dim=-1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class Int2TinyLSTM(nn.Module):
    def __init__(self, vocab_size, d_model=32, num_layers=1):
        super().__init__()
        self.embedding = Int2Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(Int2LSTMCell(d_model, d_model))
        self.fc = Int2Linear(d_model, vocab_size)
        self.d_model = d_model

    def forward(self, x, hidden=None):
        emb = self.embedding(x)
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
        out = torch.stack(outputs, dim=1)
        return self.fc(out), hidden

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_size_int2(self):
        total = 0
        for module in self.modules():
            if isinstance(module, (Int2Linear, Int2Embedding)):
                total += module.weight_fp.numel() * 2 / 8  # 2 bits per weight
                total += 4  # one float32 scale
        for name, param in self.named_parameters():
            if "bias" in name:
                total += param.numel() * 4
        return total / 1024


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
    print("INT2 Pure Training Experiment")
    print("=" * 60)

    with open(DATA_DIR / "info.json") as f:
        info = json.load(f)
    tokens = np.load(info["tokenized_file"])[:100000]
    vocab_size = info["vocab_size"]
    split = int(len(tokens) * 0.9)

    ds_train = TextDataset(tokens[:split], 64)
    ds_val = TextDataset(tokens[split:], 64)
    dl_train = DataLoader(ds_train, batch_size=4, shuffle=True, num_workers=0)
    dl_val = DataLoader(ds_val, batch_size=4, shuffle=False, num_workers=0)

    model = Int2TinyLSTM(vocab_size, d_model=32, num_layers=1)
    num_params = model.count_parameters()
    int2_size = model.get_model_size_int2()
    fp32_size = num_params * 4 / 1024

    print(f"Parameters: {num_params:,} ({num_params / 1e6:.2f}M)")
    print(f"INT2 model size: {int2_size:.1f} KB")
    print(
        f"FP32 model size: {fp32_size:.1f} KB (compression: {fp32_size / int2_size:.1f}x)"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    start = time.time()

    for epoch in range(2):
        model.train()
        total_loss, count = 0, 0
        t0 = time.time()

        for i, (x, y) in enumerate(dl_train):
            if i >= 200:
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

        model.eval()
        val_loss, val_count = 0, 0
        with torch.no_grad():
            for i, (x, y) in enumerate(dl_val):
                if i >= 200:
                    break
                logits, _ = model(x)
                val_loss += criterion(logits.view(-1, vocab_size), y.view(-1)).item()
                val_count += 1

        print(
            f"  Epoch {epoch + 1}: train={total_loss / max(count, 1):.4f}  val={val_loss / max(val_count, 1):.4f}  time={time.time() - t0:.1f}s"
        )

    final_val = val_loss / max(val_count, 1)
    total_time = time.time() - start

    print("\n--- Generation Test ---")
    model.eval()
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    prompt = "Once upon a time"
    gen = tokenizer.encode(prompt)
    ids = torch.tensor([gen], dtype=torch.long)
    with torch.no_grad():
        for _ in range(50):
            ctx = ids[:, -64:]
            logits, _ = model(ctx)
            nxt = torch.multinomial(torch.softmax(logits[0, -1, :] / 0.8, -1), 1).item()
            gen.append(nxt)
            ids = torch.tensor([gen], dtype=torch.long)
    print(f"Prompt: {prompt}")
    print(f"Output: {tokenizer.decode(gen)}")

    print("\n" + "=" * 60)
    print(f"val_loss:         {final_val:.4f}")
    print(f"training_seconds: {total_time:.1f}")
    print(f"num_params_M:     {num_params / 1e6:.2f}")
    print(f"int2_size_kb:     {int2_size:.1f}")
    print(f"compression:      {fp32_size / int2_size:.1f}x vs float32")
    print("=" * 60)


if __name__ == "__main__":
    main()
