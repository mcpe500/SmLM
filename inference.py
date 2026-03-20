#!/usr/bin/env python3
"""
inference.py - Interactive chat with trained SLM.
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"


class TinyLSTM(nn.Module):
    """Minimal LSTM language model (must match train.py)."""

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


class TinyTransformer(nn.Module):
    """Minimal Transformer language model (must match train.py)."""

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
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()
        out = self.transformer(emb, mask=mask)
        logits = self.fc(out)
        return logits, None


def load_model(arch="lstm", d_model=64, num_layers=2):
    """Load trained model."""
    from transformers import GPT2Tokenizer

    info_file = DATA_DIR / "info.json"
    with open(info_file) as f:
        info = json.load(f)
    vocab_size = info["vocab_size"]

    if arch == "lstm":
        model = TinyLSTM(vocab_size, d_model=d_model, num_layers=num_layers)
    else:
        model = TinyTransformer(
            vocab_size, d_model=d_model, nhead=2, num_layers=num_layers
        )

    model_path = MODEL_DIR / "best_model.pt"
    if not model_path.exists():
        print(f"ERROR: No trained model found at {model_path}")
        print("Run train.py first!")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return model, tokenizer


def generate(model, tokenizer, prompt, max_length=100, temperature=0.8, top_k=50):
    """Generate text from prompt."""
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    generated = list(tokens)

    with torch.no_grad():
        for _ in range(max_length):
            # Limit context to last 128 tokens for memory
            context = input_ids[:, -128:]
            logits, _ = model(context)
            next_logits = logits[0, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][-1]
                next_logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            generated.append(next_token)
            input_ids = torch.tensor([generated], dtype=torch.long)

            # Stop at EOS
            if next_token == tokenizer.eos_token_id:
                break

    return tokenizer.decode(generated)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", choices=["lstm", "transformer"], default="lstm")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt mode")
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model(args.arch, args.d_model, args.num_layers)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params:,} parameters")

    if args.prompt:
        print(f"\nPrompt: {args.prompt}")
        output = generate(
            model,
            tokenizer,
            args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
        )
        print(f"Output: {output}")
        return

    print("\n" + "=" * 50)
    print("SLM Chat - Type 'quit' to exit")
    print("=" * 50)

    while True:
        try:
            prompt = input("\nYou: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                break
            if not prompt:
                continue

            print("Model: ", end="", flush=True)
            output = generate(
                model,
                tokenizer,
                prompt,
                max_length=args.max_length,
                temperature=args.temperature,
            )
            # Remove the prompt from output
            response = output[len(tokenizer.decode(tokenizer.encode(prompt))) :].strip()
            print(response if response else "(no response)")

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
