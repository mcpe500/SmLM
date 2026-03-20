# SLM Chat on CPU - Specification Document

## Project: autoresearch-cpu-slm

**Date Started**: 2026-03-20
**Objective**: Build a Small Language Model (SLM) capable of simple chat on CPU-only hardware with strict 4GB RAM budget.

---

## 1. Hardware Constraints

| Resource | Limit | Notes |
|----------|-------|-------|
| CPU | Single Core | Will pin to 1 core via `taskset` or `OMP_NUM_THREADS=1` |
| RAM | 4GB hard limit | Must stay under 3.5GB during training (buffer for OS) |
| Disk | 30GB available | Dataset capped at 500MB |
| GPU | NONE | CUDA strictly forbidden |

**Actual Environment**: Python 3.13.7, Termux/Android, 7.1GB physical RAM (will constrain to 4GB budget).

---

## 2. Model Architecture Requirements

- **Max Parameters**: < 10 Million (strict)
- **Architecture Options** (in priority order):
  1. Tiny Transformer (2 layers, small dim)
  2. GRU-based seq2seq
  3. LSTM-based model
- **Target for baseline**: ~1M parameters (1 layer, minimal dim)
- **dtype**: `torch.float32` (CPU doesn't have efficient fp16)

---

## 3. Dataset Requirements

- **Max Size**: 500MB download
- **Candidates**:
  - TinyStories (roneneldan/TinyStories) - ~500MB, simple English stories
  - Alpaca subset - instruction following
  - Custom tiny chat dataset
- **Tokenizer**: Use pre-trained tokenizer (e.g., GPT-2 or custom BPE with small vocab)

---

## 4. Training Protocol

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch (CPU) |
| Optimizer | AdamW or Adam |
| Learning Rate | 1e-3 to 3e-4 |
| Batch Size | 1-4 (gradient accumulation to simulate larger) |
| Gradient Accumulation | 4-16 steps |
| Max Epochs | 3-5 (time-limited to 60 min per run) |
| Validation Split | 10% |

---

## 5. Output Metrics

Each run must output:
```
val_loss:         X.XXXX
training_seconds: XXXX.X
peak_ram_mb:      XXXX.X
num_params_M:     X.X
```

---

## 6. Experiment Loop Criteria

- **Keep**: val_loss decreasing AND peak_ram_mb < 3500
- **Discard**: val_loss increasing OR RAM > 3.5GB
- **Crash**: OOM or error -> analyze, fix or discard idea
- **Timeout**: Kill after 90 minutes

---

## 7. Fallback Strategy

If training from scratch fails to produce coherent output after 5 iterations:
1. Download pre-trained tiny model (e.g., Qwen-0.5B or SmolLM-135M)
2. Apply aggressive quantization (int8/dynamic)
3. Attempt light LoRA fine-tuning if RAM allows

---

## 8. Success Criteria

- Model can generate coherent English/Indonesian text
- Interactive chat produces sensible (not perfect) responses
- Training completes without OOM
- Peak RAM stays under 3.5GB
