"""Distillation training loop."""

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from .student import StudentTransformer, StudentConfig


@dataclass
class DistillationConfig:
    """Configuration for distillation training."""
    
    # Paths
    output_dir: str = "checkpoints"
    run_tag: str = "default"
    
    # Distillation hyperparameters
    temperature: float = 2.0
    alpha_kd: float = 0.7  # Weight for KL divergence
    alpha_ce: float = 0.3  # Weight for cross-entropy (if using labels)
    
    # Training hyperparameters
    learning_rate: float = 5e-5
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    num_epochs: int = 10
    max_steps: Optional[int] = None
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    
    # Checkpointing
    save_steps: int = 5000
    logging_steps: int = 100
    
    # Device
    device: str = "cpu"
    
    @classmethod
    def from_yaml(cls, path: str) -> "DistillationConfig":
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def save(self, path: str):
        """Save config to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.__dict__, f)


class DistillationTrainer:
    """Trainer for knowledge distillation."""
    
    def __init__(
        self,
        student: StudentTransformer,
        teacher: Any,  # Teacher model
        config: DistillationConfig,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ):
        self.student = student
        self.teacher = teacher
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        self.device = torch.device(config.device)
        self.student = self.student.to(self.device)
        self.teacher = self.teacher.to(self.device)
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=config.learning_rate,
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Output directory
        self.output_dir = Path(config.output_dir) / config.run_tag
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Compute KL divergence loss for distillation."""
        
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        kd_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
        )
        
        return kd_loss * (temperature ** 2)
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        
        self.student.train()
        
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Teacher forward (no grad)
        with torch.no_grad():
            if hasattr(self.teacher, 'transformer'):
                teacher_out = self.teacher.transformer(input_ids)
            else:
                teacher_out = self.teacher(input_ids)
            
            if isinstance(teacher_out, tuple):
                teacher_logits = teacher_out[0]
            else:
                teacher_logits = teacher_out.logits if hasattr(teacher_out, 'logits') else teacher_out
        
        # Student forward
        student_logits = self.student(input_ids)
        
        # Align dimensions if needed
        if student_logits.shape[-1] != teacher_logits.shape[-1]:
            # Vocab size mismatch - truncate to smaller
            min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
            student_logits = student_logits[..., :min_vocab]
            teacher_logits = teacher_logits[..., :min_vocab]
        
        # Compute loss
        loss = self.compute_distillation_loss(
            student_logits,
            teacher_logits,
            self.config.temperature,
        )
        
        # Backward pass with gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single evaluation step."""
        
        self.student.eval()
        
        input_ids = batch['input_ids'].to(self.device)
        
        with torch.no_grad():
            # Teacher forward
            with torch.no_grad():
                if hasattr(self.teacher, 'transformer'):
                    teacher_out = self.teacher.transformer(input_ids)
                else:
                    teacher_out = self.teacher(input_ids)
                
                if isinstance(teacher_out, tuple):
                    teacher_logits = teacher_out[0]
                else:
                    teacher_logits = teacher_out.logits if hasattr(teacher_out, 'logits') else teacher_out
            
            # Student forward
            student_logits = self.student(input_ids)
            
            # Align dimensions if needed
            if student_logits.shape[-1] != teacher_logits.shape[-1]:
                min_vocab = min(student_logits.shape[-1], teacher_logits.shape[-1])
                student_logits = student_logits[..., :min_vocab]
                teacher_logits = teacher_logits[..., :min_vocab]
            
            # Compute loss
            loss = self.compute_distillation_loss(
                student_logits,
                teacher_logits,
                self.config.temperature,
            )
        
        return loss.item()
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        
        # Resume from checkpoint if specified
        if resume_from:
            self._load_checkpoint(resume_from)
        
        total_steps = self.config.max_steps or (
            len(self.train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps
        )
        
        start_time = time.time()
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            
            pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
            total_loss = 0.0
            steps_this_epoch = 0
            
            self.optimizer.zero_grad()
            
            for batch in pbar:
                loss = self.train_step(batch)
                total_loss += loss
                steps_this_epoch += 1
                
                # Gradient step
                if (self.global_step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.student.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / max(1, steps_this_epoch)
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
                
                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint()
                
                # Max steps reached
                if self.global_step >= total_steps:
                    break
            
            # End of epoch evaluation
            if self.eval_dataloader:
                eval_loss = self.evaluate()
                if eval_loss < self.best_loss:
                    self.best_loss = eval_loss
                    self._save_checkpoint(suffix="best")
            
            # Save epoch checkpoint
            self._save_checkpoint(suffix=f"epoch-{epoch}")
        
        # Final save
        self._save_checkpoint()
        
        total_time = time.time() - start_time
        print(f"Training completed in {total_time:.2f}s")
        
        return self.best_loss
    
    def evaluate(self) -> float:
        """Run evaluation."""
        
        if not self.eval_dataloader:
            return 0.0
        
        self.student.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                loss = self.eval_step(batch)
                total_loss += loss
                num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def _save_checkpoint(self, suffix: str = ""):
        """Save checkpoint."""
        
        checkpoint_name = f"checkpoint-{self.global_step}"
        if suffix:
            checkpoint_name = f"{checkpoint_name}-{suffix}"
        
        checkpoint_path = self.output_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        torch.save({
            'step': self.global_step,
            'epoch': self.epoch,
            'student_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config,
        }, checkpoint_path / "checkpoint.pt")
        
        # Save config
        self.config.save(checkpoint_path / "config.yaml")
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def _load_checkpoint(self, path: str):
        """Load checkpoint."""
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.global_step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.best_loss = checkpoint['best_loss']
        
        self.student.load_state_dict(checkpoint['student_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"Loaded checkpoint: {path}")
