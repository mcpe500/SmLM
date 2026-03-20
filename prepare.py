#!/usr/bin/env python3
"""
prepare.py - Download and prepare dataset for SLM training.
Downloads a small subset of TinyStories and sets up tokenizer.
"""

import os
import sys
import json
import hashlib
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def get_tokenizer():
    """Return GPT-2 tokenizer from transformers."""
    from transformers import GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    return tokenizer


def download_tinystories(max_size_mb=20):
    """Download TinyStories dataset, limit to max_size_mb."""
    from datasets import load_dataset

    cache_file = DATA_DIR / "tinystories_subset.jsonl"
    if cache_file.exists():
        print(f"Dataset already exists at {cache_file}")
        return cache_file

    print("Downloading TinyStories dataset...")
    # Load only the train split, streaming to control memory
    ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True)

    total_bytes = 0
    max_bytes = max_size_mb * 1024 * 1024
    count = 0

    with open(cache_file, "w") as f:
        for example in ds:
            text = example["text"]
            line = json.dumps({"text": text}) + "\n"
            total_bytes += len(line.encode())
            f.write(line)
            count += 1
            if count % 1000 == 0:
                print(
                    f"  Downloaded {count} stories ({total_bytes / 1024 / 1024:.1f} MB)"
                )
            if total_bytes >= max_bytes:
                print(f"  Reached size limit ({max_size_mb} MB)")
                break

    print(f"Downloaded {count} stories ({total_bytes / 1024 / 1024:.1f} MB)")
    return cache_file


def prepare_tokenized_data(data_file, tokenizer, max_length=128, max_tokens=2_000_000):
    """Tokenize dataset and save as numpy arrays for fast loading."""
    import numpy as np

    tokenized_file = DATA_DIR / "tokenized.npy"
    if tokenized_file.exists():
        print(f"Tokenized data already exists at {tokenized_file}")
        return tokenized_file

    print("Tokenizing dataset...")
    all_tokens = []

    with open(data_file) as f:
        for i, line in enumerate(f):
            example = json.loads(line)
            text = example["text"]
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            all_tokens.extend(tokens)
            if (i + 1) % 1000 == 0:
                print(f"  Tokenized {i + 1} stories ({len(all_tokens)} tokens)")
            if len(all_tokens) >= max_tokens:
                print(f"  Reached token limit ({max_tokens})")
                break

    tokens_array = np.array(all_tokens, dtype=np.uint16)
    np.save(tokenized_file, tokens_array)
    print(f"Saved {len(tokens_array)} tokens to {tokenized_file}")
    return tokenized_file


def main():
    print("=" * 50)
    print("SLM Data Preparation")
    print("=" * 50)

    # Check available RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemAvailable" in line:
                    available_kb = int(line.split()[1])
                    print(f"Available RAM: {available_kb / 1024:.0f} MB")
                    if available_kb < 1024 * 1024:  # < 1GB
                        print("WARNING: Less than 1GB RAM available!")
                    break
    except:
        pass

    # Step 1: Get tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = get_tokenizer()
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Step 2: Download dataset
    print("\n[2/3] Downloading dataset...")
    data_file = download_tinystories(max_size_mb=20)

    # Step 3: Tokenize
    print("\n[3/3] Tokenizing...")
    tokenized_file = prepare_tokenized_data(
        data_file, tokenizer, max_length=128, max_tokens=2_000_000
    )

    # Save tokenizer info
    info = {
        "vocab_size": tokenizer.vocab_size,
        "max_length": 128,
        "data_file": str(data_file),
        "tokenized_file": str(tokenized_file),
    }
    with open(DATA_DIR / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 50)
    print("Data preparation complete!")
    print(f"  Dataset: {data_file}")
    print(f"  Tokenized: {tokenized_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()
