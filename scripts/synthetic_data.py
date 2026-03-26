#!/usr/bin/env python3
"""Generate synthetic data for smoke testing without network access."""

import torch
from pathlib import Path


def generate_synthetic_dataset(
    vocab_size: int = 50257,
    num_samples: int = 1000,
    max_seq_len: int = 512,
    output_path: str = "data/synthetic_train.pt",
):
    """Generate synthetic tokenized data for testing."""
    
    print(f"Generating synthetic dataset...")
    print(f"  - Vocab size: {vocab_size}")
    print(f"  - Samples: {num_samples}")
    print(f"  - Max seq len: {max_seq_len}")
    
    data = []
    for i in range(num_samples):
        # Variable length sequences
        length = torch.randint(32, max_seq_len, (1,)).item()
        # Use realistic token distribution (more common tokens)
        tokens = torch.randint(100, min(10000, vocab_size), (length,))
        data.append(tokens)
    
    # Split train/eval
    split_idx = int(len(data) * 0.95)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'train': train_data,
        'eval': eval_data,
    }, output_path)
    
    print(f"Saved to {output_path}")
    print(f"  - Train samples: {len(train_data)}")
    print(f"  - Eval samples: {len(eval_data)}")
    
    return train_data, eval_data


def load_synthetic_dataset(path: str = "data/synthetic_train.pt"):
    """Load synthetic dataset."""
    
    path = Path(path)
    if not path.exists():
        return generate_synthetic_dataset()
    
    data = torch.load(path)
    return data['train'], data['eval']


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic dataset')
    parser.add_argument('--vocab-size', type=int, default=50257)
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--max-seq-len', type=int, default=512)
    parser.add_argument('--output', type=str, default='data/synthetic_train.pt')
    
    args = parser.parse_args()
    
    generate_synthetic_dataset(
        vocab_size=args.vocab_size,
        num_samples=args.num_samples,
        max_seq_len=args.max_seq_len,
        output_path=args.output,
    )
