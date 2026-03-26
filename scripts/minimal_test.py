#!/usr/bin/env python3
"""Minimal smoke test - works without transformers/datasets packages."""

import sys
import os
import time
import torch
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from compressor.student import StudentConfig, StudentTransformer
from compressor.distill import DistillationConfig, DistillationTrainer
from torch.utils.data import DataLoader


def load_synthetic_data(path: str = "data/synthetic_train.pt"):
    """Load synthetic dataset."""
    data = torch.load(path, weights_only=False)
    return data['train'], data['eval']


def create_dataloader(data, batch_size: int = 16, pad_token_id: int = 0):
    """Create dataloader from tokenized data."""
    
    def collate_fn(batch):
        max_len = max(len(x) for x in batch)
        padded = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
        for i, x in enumerate(batch):
            padded[i, :len(x)] = x
        return {'input_ids': padded}
    
    return DataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


def create_mock_teacher(vocab_size: int, hidden_size: int):
    """Create a simple mock teacher model."""
    
    class MockTeacher(torch.nn.Module):
        def __init__(self, vocab_size, hidden_size):
            super().__init__()
            self.embed = torch.nn.Embedding(vocab_size, hidden_size)
            self.proj = torch.nn.Linear(hidden_size, vocab_size)
        
        def forward(self, input_ids):
            x = self.embed(input_ids)
            logits = self.proj(x)
            return logits
    
    return MockTeacher(vocab_size, hidden_size)


def run_smoke_test():
    """Run minimal smoke test."""
    
    print("=" * 60)
    print("SmLM Minimal Smoke Test")
    print("=" * 60)
    print()
    
    # Load config
    print("Loading config: configs/minimal-test.yaml")
    with open('configs/minimal-test.yaml', 'r') as f:
        config_data = yaml.safe_load(f)
    
    student_cfg_data = config_data.get('student_config', {})
    training_cfg_data = config_data.get('training', {})
    
    # Create student config
    student_config = StudentConfig(
        vocab_size=student_cfg_data.get('vocab_size', 50257),
        max_position_embeddings=student_cfg_data.get('max_position_embeddings', 1024),
        num_hidden_layers=student_cfg_data.get('num_hidden_layers', 2),
        num_attention_heads=student_cfg_data.get('num_attention_heads', 4),
        hidden_size=student_cfg_data.get('hidden_size', 128),
        intermediate_size=student_cfg_data.get('intermediate_size', 512),
    )
    
    print(f"Student config:")
    print(f"  - Layers: {student_config.num_hidden_layers}")
    print(f"  - Heads: {student_config.num_attention_heads}")
    print(f"  - Hidden: {student_config.hidden_size}")
    print(f"  - FFN: {student_config.intermediate_size}")
    print()
    
    # Create models
    print("Creating student model...")
    student = StudentTransformer(student_config)
    student_params = sum(p.numel() for p in student.parameters())
    print(f"  - Student parameters: {student_params:,} ({student_params/1e6:.3f}M)")
    
    print("Creating mock teacher...")
    teacher = create_mock_teacher(
        student_config.vocab_size,
        student_config.hidden_size,
    )
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"  - Teacher parameters: {teacher_params:,}")
    print()
    
    # Load data
    print("Loading synthetic data...")
    train_data, eval_data = load_synthetic_data()
    print(f"  - Train samples: {len(train_data)}")
    print(f"  - Eval samples: {len(eval_data)}")
    
    train_loader = create_dataloader(train_data, batch_size=training_cfg_data.get('batch_size', 16))
    eval_loader = create_dataloader(eval_data, batch_size=training_cfg_data.get('batch_size', 16))
    print()
    
    # Create distillation config
    distill_config = DistillationConfig(
        output_dir='checkpoints',
        run_tag='smoke-test-' + str(int(time.time())),
        temperature=float(config_data.get('distillation', {}).get('temperature', 2.0)),
        learning_rate=float(training_cfg_data.get('learning_rate', 1e-4)),
        batch_size=int(training_cfg_data.get('batch_size', 16)),
        gradient_accumulation_steps=int(training_cfg_data.get('gradient_accumulation_steps', 2)),
        num_epochs=int(training_cfg_data.get('num_epochs', 1)),
        warmup_steps=int(training_cfg_data.get('warmup_steps', 100)),
        save_steps=int(training_cfg_data.get('save_steps', 50)),
        logging_steps=int(training_cfg_data.get('logging_steps', 10)),
        device='cpu',
    )
    
    # Create trainer
    print("Creating trainer...")
    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        config=distill_config,
        train_dataloader=train_loader,
        eval_dataloader=eval_loader,
    )
    print()
    
    # Train
    print("Starting training...")
    start_time = time.time()
    
    best_loss = trainer.train()
    
    elapsed = time.time() - start_time
    print()
    print(f"Training completed in {elapsed:.1f}s")
    print(f"Best eval loss: {best_loss:.4f}")
    print()
    
    # Test generation
    print("Testing generation...")
    student.eval()
    input_ids = torch.randint(100, 1000, (1, 16))
    with torch.no_grad():
        generated = student.generate(input_ids, max_new_tokens=20)
    print(f"  - Input: {list(input_ids.shape)}")
    print(f"  - Output: {list(generated.shape)}")
    print()
    
    # Summary
    print("=" * 60)
    print("SMOKE TEST PASSED ✅")
    print("=" * 60)
    print()
    print("Results:")
    print(f"  - Model trained: {student_params:,} parameters")
    print(f"  - Training time: {elapsed:.1f}s")
    print(f"  - Best loss: {best_loss:.4f}")
    print(f"  - Checkpoint: {trainer.output_dir}")
    print()
    
    return True


if __name__ == '__main__':
    success = run_smoke_test()
    sys.exit(0 if success else 1)
