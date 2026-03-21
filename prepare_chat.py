#!/usr/bin/env python3
"""
prepare_chat.py - Download SmolLM2-135M-Instruct and chat dataset.
Prepares everything for fine-tuning on CPU with 4GB RAM constraint.
"""

import os
import sys
import json
import time
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
CHAT_DIR = DATA_DIR / "chat"
CHAT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"


def get_model():
    """Download SmolLM2-135M-Instruct model and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "HuggingFaceTB/SmolLM2-135M-Instruct"
    local_model_dir = MODEL_DIR / "smollm2-135m-instruct"

    if local_model_dir.exists() and (local_model_dir / "config.json").exists():
        print(f"Model already exists at {local_model_dir}")
        tokenizer = AutoTokenizer.from_pretrained(str(local_model_dir))
        return str(local_model_dir), tokenizer

    print(f"Downloading model: {model_id}")
    print("This may take a few minutes...")

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.save_pretrained(str(local_model_dir))
    print(f"Tokenizer saved to {local_model_dir}")
    print(f"Time: {time.time() - t0:.1f}s")

    print(
        "Note: Model weights will be loaded directly from HuggingFace during training."
    )
    print(f"Model ID: {model_id}")
    print(f"Local config saved at: {local_model_dir}")

    return model_id, tokenizer


def get_chat_dataset():
    """Download and prepare chat dataset for fine-tuning."""
    from datasets import load_dataset

    dataset_id = "HuggingFaceTB/everyday-conversations-llama3.1-2k"
    local_file = CHAT_DIR / "train_chat.jsonl"

    if local_file.exists():
        print(f"Dataset already exists at {local_file}")
        return local_file

    print(f"\nDownloading dataset: {dataset_id}")
    t0 = time.time()

    ds = load_dataset(dataset_id, split="train_sft")
    print(f"Loaded {len(ds)} examples")

    with open(local_file, "w") as f:
        for example in ds:
            row = {"messages": example["messages"]}
            f.write(json.dumps(row) + "\n")

    print(f"Saved {len(ds)} examples to {local_file}")
    print(f"Time: {time.time() - t0:.1f}s")
    print(f"File size: {local_file.stat().st_size / 1024 / 1024:.2f} MB")

    return local_file


def get_instruction_dataset():
    """Download Alpaca-style instruction dataset as backup."""
    from datasets import load_dataset

    dataset_id = "yahma/alpaca-cleaned"
    local_file = CHAT_DIR / "alpaca_train.jsonl"

    if local_file.exists():
        print(f"Alpaca dataset already exists at {local_file}")
        return local_file

    print(f"\nDownloading Alpaca dataset: {dataset_id}")
    t0 = time.time()

    ds = load_dataset(dataset_id, split="train")
    print(f"Loaded {len(ds)} examples")

    with open(local_file, "w") as f:
        for example in ds:
            messages = [
                {
                    "role": "user",
                    "content": example["instruction"] + "\n" + example.get("input", ""),
                },
                {"role": "assistant", "content": example["output"]},
            ]
            row = {"messages": messages}
            f.write(json.dumps(row) + "\n")

    print(f"Saved {len(ds)} examples to {local_file}")
    print(f"Time: {time.time() - t0:.1f}s")

    return local_file


def prepare_training_data(chat_file, tokenizer):
    """Prepare tokenized training data from chat format."""
    import torch
    from torch.utils.data import Dataset

    print(f"\nPreparing training data from {chat_file}")

    class ChatDataset(Dataset):
        def __init__(self, file_path, tokenizer, max_length=256):
            self.examples = []
            self.tokenizer = tokenizer
            self.max_length = max_length

            with open(file_path) as f:
                for line in f:
                    self.examples.append(json.loads(line))

            print(f"Loaded {len(self.examples)} chat examples")

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            example = self.examples[idx]
            messages = example["messages"]

            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )

            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )

            input_ids = encoding["input_ids"].squeeze()
            attention_mask = encoding["attention_mask"].squeeze()

            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }

    return ChatDataset(chat_file, tokenizer)


def save_info(model_id, tokenizer, chat_file, alpaca_file):
    """Save preparation info."""
    info = {
        "model_id": model_id,
        "vocab_size": tokenizer.vocab_size,
        "chat_dataset": str(chat_file),
        "alpaca_dataset": str(alpaca_file),
        "chat_template": tokenizer.chat_template or "smollm2",
    }

    with open(CHAT_DIR / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nInfo saved to {CHAT_DIR / 'info.json'}")


def main():
    print("=" * 60)
    print("Chat Model Preparation - CPU Constrained")
    print("=" * 60)

    print("\nChecking available RAM...")
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if "MemAvailable" in line:
                    avail_mb = int(line.split()[1]) / 1024
                    print(f"Available RAM: {avail_mb:.0f} MB")
                    break
    except:
        pass

    print("\n[1/4] Downloading model...")
    model_id, tokenizer = get_model()

    print("\n[2/4] Downloading chat dataset...")
    chat_file = get_chat_dataset()

    print("\n[3/4] Downloading Alpaca dataset...")
    alpaca_file = get_instruction_dataset()

    print("\n[4/4] Saving preparation info...")
    save_info(model_id, tokenizer, chat_file, alpaca_file)

    print("\n" + "=" * 60)
    print("Preparation complete!")
    print(f"  Model: {model_id}")
    print(f"  Chat dataset: {chat_file}")
    print(f"  Alpaca dataset: {alpaca_file}")
    print(
        f"  Chat template: {tokenizer.chat_template[:100] if tokenizer.chat_template else 'default'}..."
    )
    print("=" * 60)

    print("\nNext step: Run train_finetune.py to start training")


if __name__ == "__main__":
    main()
