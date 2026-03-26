"""Quality evaluation harness."""

import torch
import torch.nn.functional as F
from typing import List, Optional, Any
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Evaluation result."""
    
    perplexity: float
    loss: float
    num_tokens: int
    num_batches: int


def compute_perplexity(
    model: Any,
    dataloader: Any,
    device: str = "cpu",
    max_batches: Optional[int] = None,
) -> EvalResult:
    """Compute perplexity on a dataset."""
    
    model.eval()
    model = model.to(device)
    
    total_loss = 0.0
    total_tokens = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if max_batches and num_batches >= max_batches:
                break
            
            input_ids = batch['input_ids'].to(device)
            labels = input_ids.clone()
            
            # Forward pass
            outputs = model(input_ids)
            
            # Compute loss
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Shift for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            
            total_loss += loss.item() * shift_labels.numel()
            total_tokens += shift_labels.numel()
            num_batches += 1
    
    avg_loss = total_loss / max(1, total_tokens)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return EvalResult(
        perplexity=round(perplexity, 4),
        loss=round(avg_loss, 6),
        num_tokens=total_tokens,
        num_batches=num_batches,
    )


def compute_bits_per_byte(perplexity: float) -> float:
    """Convert perplexity to bits per byte."""
    
    import math
    return math.log2(perplexity) / 8


def evaluate_generation(
    model: Any,
    tokenizer: Any,
    prompts: List[str],
    max_new_tokens: int = 50,
    device: str = "cpu",
) -> List[str]:
    """Evaluate model generation quality."""
    
    model.eval()
    model = model.to(device)
    
    generations = []
    
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors='pt').to(device)
            
            if hasattr(model, 'generate'):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )
            else:
                # Custom generate for StudentTransformer
                input_ids = inputs['input_ids']
                outputs = model.generate(input_ids, max_new_tokens=max_new_tokens)
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generations.append(generated)
    
    return generations


def compare_models(
    teacher: Any,
    student: Any,
    dataloader: Any,
    device: str = "cpu",
) -> dict:
    """Compare teacher and student models."""
    
    print("Evaluating teacher...")
    teacher_result = compute_perplexity(teacher, dataloader, device)
    
    print("Evaluating student...")
    student_result = compute_perplexity(student, dataloader, device)
    
    # Compute retention
    teacher_bpb = compute_bits_per_byte(teacher_result.perplexity)
    student_bpb = compute_bits_per_byte(student_result.perplexity)
    
    bpb_diff = student_bpb - teacher_bpb
    retention = 1 - (bpb_diff / teacher_bpb) if teacher_bpb > 0 else 0
    
    return {
        'teacher': {
            'perplexity': teacher_result.perplexity,
            'loss': teacher_result.loss,
            'bits_per_byte': round(teacher_bpb, 4),
        },
        'student': {
            'perplexity': student_result.perplexity,
            'loss': student_result.loss,
            'bits_per_byte': round(student_bpb, 4),
        },
        'comparison': {
            'perplexity_ratio': student_result.perplexity / teacher_result.perplexity,
            'bpb_diff': round(bpb_diff, 4),
            'retention': round(retention * 100, 2),
        }
    }
