#!/usr/bin/env python3
"""Distillation training script."""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from compressor.student import create_student_from_teacher, StudentConfig
from compressor.distill import DistillationTrainer, DistillationConfig


def load_dataset(tokenizer, max_seq_len: int = 512, num_samples: int = 10000):
    """Load and tokenize dataset."""
    
    print("Loading dataset...")
    
    try:
        from datasets import load_dataset
        dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    except Exception as e:
        print(f"Could not load wikitext: {e}")
        print("Using synthetic data for smoke test...")
        
        # Synthetic data for smoke testing
        data = []
        for i in range(num_samples):
            length = torch.randint(32, max_seq_len, (1,)).item()
            tokens = torch.randint(100, 1000, (length,))
            data.append(tokens)
        
        return data
    
    # Tokenize
    texts = dataset['text'][:num_samples]
    tokenized = tokenizer(texts, max_length=max_seq_len, truncation=True, padding=False)
    
    # Convert to list of tensors
    data = [torch.tensor(ids) for ids in tokenized['input_ids']]
    
    return data


def create_dataloader(data, batch_size: int = 16, pad_token_id: int = 0):
    """Create dataloader from tokenized data."""
    
    # Pad and batch
    def collate_fn(batch):
        max_len = max(len(x) for x in batch)
        padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        for i, x in enumerate(batch):
            padded[i, :len(x)] = x
        return {'input_ids': padded}
    
    return DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def main():
    parser = argparse.ArgumentParser(description='Distillation training')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--smoke-test', action='store_true', help='Run quick smoke test')
    args = parser.parse_args()
    
    # Load config
    config = DistillationConfig.from_yaml(args.config)
    
    if args.smoke_test:
        config.num_epochs = 1
        config.save_steps = 100
        config.logging_steps = 10
        config.run_tag = config.run_tag + "-smoke"
        print("Running smoke test mode")
    
    print(f"Run tag: {config.run_tag}")
    print(f"Output dir: {config.output_dir}")
    
    # Load teacher model
    print("Loading teacher model...")
    teacher_name = "HuggingFaceTB/SmolLM2-360M"
    
    if args.smoke_test:
        print("Using GPT-2 for smoke test...")
        teacher_name = "gpt2"
    
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name)
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Teacher: {teacher_name}")
    print(f"  - Parameters: {sum(p.numel() for p in teacher.parameters()) / 1e6:.2f}M")
    
    # Create student
    print("Creating student model...")
    shrink_factor = 0.5 if args.smoke_test else 0.4
    layer_ratio = 0.5 if args.smoke_test else 0.4
    
    student, student_config = create_student_from_teacher(
        teacher,
        shrink_factor=shrink_factor,
        layer_ratio=layer_ratio,
    )
    
    print(f"Student:")
    print(f"  - Layers: {student_config.num_hidden_layers}")
    print(f"  - Heads: {student_config.num_attention_heads}")
    print(f"  - Hidden: {student_config.hidden_size}")
    print(f"  - FFN: {student_config.intermediate_size}")
    print(f"  - Parameters: {sum(p.numel() for p in student.parameters()) / 1e6:.2f}M")
    
    # Load dataset
    data = load_dataset(
        tokenizer,
        max_seq_len=config.max_seq_len if hasattr(config, 'max_seq_len') else 512,
        num_samples=1000 if args.smoke_test else 10000,
    )
    
    # Split train/eval
    split_idx = int(len(data) * 0.95)
    train_data = data[:split_idx]
    eval_data = data[split_idx:]
    
    print(f"Dataset: {len(train_data)} train, {len(eval_data)} eval")
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_data,
        batch_size=config.batch_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    eval_loader = create_dataloader(
        eval_data,
        batch_size=config.batch_size,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Create trainer
    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        config=config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
    )
    
    # Train
    print("Starting training...")
    best_loss = trainer.train(resume_from=args.resume)
    
    print(f"Training complete! Best eval loss: {best_loss:.4f}")
    
    # Save final model
    final_path = Path(config.output_dir) / config.run_tag / "final"
    final_path.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'student_state_dict': student.state_dict(),
        'student_config': student_config,
    }, final_path / "model.pt")
    
    print(f"Saved final model to {final_path}")


if __name__ == '__main__':
    main()
