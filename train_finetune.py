#!/usr/bin/env python3
"""
train_finetune.py - LoRA fine-tuning of SmolLM2-135M-Instruct on CPU.
Constrained to 4GB RAM with aggressive memory optimization.

Strategy:
- Load model in low precision (FP16 with memory optimization, or INT8 if available)
- Apply LoRA adapters (freeze base)
- Train only LoRA params (minimal memory)
- Gradient accumulation for effective batch
"""

import os
import sys
import time
import json
import resource
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)

DATA_DIR = Path(__file__).parent / "data"
MODEL_DIR = Path(__file__).parent / "models"
CHAT_DIR = DATA_DIR / "chat"
OUTPUT_DIR = MODEL_DIR / "finetuned"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TrainingConfig:
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    dataset: str = "chat"
    max_length: int = 128
    batch_size: int = 1
    grad_accum_steps: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    max_steps: int = 500
    eval_steps: int = 100
    save_steps: int = 200
    warmup_steps: int = 10
    lora_rank: int = 4
    lora_alpha: int = 8
    lora_dropout: float = 0.05
    target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")
    dtype: str = "float16"


def get_peak_ram_mb():
    """Get peak RAM usage in MB."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / 1024


class ChatDataset(Dataset):
    """Dataset for chat fine-tuning."""

    def __init__(self, file_path, tokenizer, max_length=128, split="train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        print(f"Loading dataset from {file_path}")
        with open(file_path) as f:
            for line in f:
                self.examples.append(json.loads(line))

        print(f"Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        messages = example["messages"]

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            text = self._fallback_format(messages)

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _fallback_format(self, messages):
        """Fallback if chat template fails."""
        text = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        text += "<|im_start|>assistant\n"
        return text


def load_model_and_tokenizer(config):
    """Load model with memory optimization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print(f"\nLoading model: {config.model_id}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>' + messages[-1]['role'] + '\n' }}"
            "{% endif %}"
        )

    print(f"Tokenizer loaded in {time.time() - t0:.1f}s")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(config.dtype, torch.float16)

    print(f"Loading model in {dtype} with low memory usage...")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=None,
        trust_remote_code=True,
    )

    model.config.use_cache = False

    print(f"Model loaded in {time.time() - t0:.1f}s")
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    return model, tokenizer


def apply_lora(model, config):
    """Apply LoRA to model."""
    from peft import LoraConfig, get_peft_model, TaskType

    print(f"\nApplying LoRA (rank={config.lora_rank}, alpha={config.lora_alpha})")

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
    )

    model = get_peft_model(model, lora_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"Trainable params: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    print(f"Total params: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Frozen params: {(total_params - trainable_params) / 1e6:.2f}M")

    return model


def createoptimizer(model, config):
    """Create optimizer for LoRA params only."""
    optimizer_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            optimizer_params.append(
                {"params": [param], "weight_decay": config.weight_decay}
            )

    optimizer = AdamW(optimizer_params, lr=config.learning_rate)
    return optimizer


def train_step(model, batch, config):
    """Single training step."""
    input_ids = batch["input_ids"].unsqueeze(0)
    attention_mask = batch["attention_mask"].unsqueeze(0)
    labels = batch["labels"].unsqueeze(0)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    loss = outputs.loss / config.grad_accum_steps
    loss.backward()

    return loss.item() * config.grad_accum_steps


def evaluate(model, dataloader, config):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    num_examples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].unsqueeze(0)
            attention_mask = batch["attention_mask"].unsqueeze(0)
            labels = batch["labels"].unsqueeze(0)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            total_loss += outputs.loss.item()
            num_examples += 1

            if num_examples >= 50:
                break

    model.train()
    return total_loss / max(num_examples, 1)


def save_model(model, tokenizer, output_dir, config):
    """Save fine-tuned model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    info = {
        "config": {
            "lora_rank": config.lora_rank,
            "lora_alpha": config.lora_alpha,
            "max_length": config.max_length,
        },
        "metrics": {},
    }
    with open(output_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Model saved to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=TrainingConfig.model_id)
    parser.add_argument("--dataset", choices=["chat", "alpaca"], default="chat")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float16"
    )
    args = parser.parse_args()

    config = TrainingConfig(
        model_id=args.model_id,
        dataset=args.dataset,
        max_length=args.max_length,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        dtype=args.dtype,
    )

    print("=" * 60)
    print("LoRA Fine-tuning - CPU Constrained")
    print("=" * 60)
    print(f"Model: {config.model_id}")
    print(f"Dataset: {config.dataset}")
    print(f"Max length: {config.max_length}")
    print(f"Batch size: {config.batch_size}")
    print(f"Grad accum: {config.grad_accum_steps}")
    print(f"LoRA rank: {config.lora_rank}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max steps: {config.max_steps}")
    print(f"dtype: {config.dtype}")
    print(f"Initial RAM: {get_peak_ram_mb():.0f} MB")

    dataset_file = CHAT_DIR / (
        "alpaca_train.jsonl" if config.dataset == "alpaca" else "train_chat.jsonl"
    )
    if not dataset_file.exists():
        print(f"\nERROR: Dataset not found at {dataset_file}")
        print("Run prepare_chat.py first!")
        sys.exit(1)

    print(f"\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config)

    print("\nApplying LoRA...")
    model = apply_lora(model, config)

    print("\nLoading dataset...")
    dataset = ChatDataset(dataset_file, tokenizer, config.max_length)

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x: x[0]
    )

    optimizer = createoptimizer(model, config)

    print(f"\nStarting training...")
    print(f"Effective batch size: {config.batch_size * config.grad_accum_steps}")
    print(f"Updates per epoch: ~{len(dataset) // config.grad_accum_steps}")

    start_time = time.time()
    model.train()

    total_loss = 0
    step_count = 0
    grad_step = 0
    best_val_loss = float("inf")
    peak_ram = get_peak_ram_mb()

    dataloader_iter = iter(dataloader)

    while step_count < config.max_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        loss = train_step(model, batch, config)
        total_loss += loss
        grad_step += 1

        if grad_step % config.grad_accum_steps == 0:
            step_count += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            avg_loss = total_loss / step_count
            current_ram = get_peak_ram_mb()
            peak_ram = max(peak_ram, current_ram)
            elapsed = time.time() - start_time

            print(
                f"Step {step_count}/{config.max_steps} | "
                f"loss={avg_loss:.4f} | "
                f"RAM={current_ram:.0f}MB | "
                f"time={elapsed:.0f}s"
            )

            if step_count % config.eval_steps == 0:
                val_loss = evaluate(model, dataloader, config)
                print(f"  Val loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f"  -> Best model! (val_loss={val_loss:.4f})")

            if step_count % config.save_steps == 0:
                save_model(model, tokenizer, OUTPUT_DIR / f"step_{step_count}", config)

            if current_ram > 3500:
                print(f"\nWARNING: RAM exceeds 3.5GB limit! ({current_ram:.0f}MB)")
                break

            if elapsed > 3600:
                print("\nTimeout reached (60 min)")
                break

    total_time = time.time() - start_time
    final_ram = get_peak_ram_mb()

    save_model(model, tokenizer, OUTPUT_DIR / "final", config)

    print("\n" + "=" * 60)
    print(f"val_loss:         {best_val_loss:.4f}")
    print(f"training_seconds: {total_time:.1f}")
    print(f"peak_ram_mb:      {peak_ram:.1f}")
    print(f"num_steps:        {step_count}")
    print(f"lora_rank:        {config.lora_rank}")
    print("=" * 60)


if __name__ == "__main__":
    main()
