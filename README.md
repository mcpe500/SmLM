# autoresearch-cpu-slm

Autonomous experiment to build a Small Language Model (SLM) on CPU-only hardware with strict 4GB RAM constraint. No GPU. No CUDA. No excuses.

## Hardware Constraints

| Resource | Limit |
|----------|-------|
| CPU | Single Core |
| RAM | 4GB (3.5GB usable) |
| GPU | None |
| Dataset | < 500MB |

## Project Structure

```
prepare.py          Dataset download (TinyStories) and tokenization
train.py            Training loop (LSTM + Transformer, FP32)
inference.py        Interactive chat / text generation (legacy)
int4_train_test.py  Pure INT4 training with STE (7.2x compression)
int2_train_test.py  Pure INT2 training with STE (13.0x compression)
prepare_chat.py     Download SmolLM2-135M-Instruct + chat dataset
train_finetune.py   LoRA fine-tuning (slow on CPU)
inference_chat.py   Multi-turn interactive chat (RECOMMENDED)
results.tsv         Experiment log
pyproject.toml      Dependencies
program.md          Experiment protocol and rules
venv2/              Python environment with all dependencies
data/
  chat/
    train_chat.jsonl    Everyday conversations dataset (2.2 MB)
    alpaca_train.jsonl  Alpaca instruction dataset (43 MB)
```

## Quick Start

```bash
# Activate environment
export PATH="/root/SmLM/venv2/bin:$PATH"

# Chat with pre-trained SmolLM2-135M-Instruct (WORKS NOW!)
python inference_chat.py --model-path HuggingFaceTB/SmolLM2-135M-Instruct

# Interactive chat with commands:
#   /quit          - Exit
#   /clear         - Clear conversation history
#   /history       - Show conversation history
#   /system <text> - Set system prompt

# Single prompt mode:
python inference_chat.py --model-path HuggingFaceTB/SmolLM2-135M-Instruct \
    --prompt "Hello, how are you?" --max-new-tokens 50
```

## Legacy: Training from Scratch

The original experiment trained LSTM models from scratch. These still work but produce gibberish due to insufficient training iterations.

```bash
# Prepare TinyStories dataset
python prepare.py

# Train baseline (200 steps, ~8 min)
python train.py --arch lstm --d-model 32 --num-layers 1 --epochs 2 --max-steps 200

# Chat with trained model
python inference.py --arch lstm --d-model 32 --num-layers 1 --prompt "Hello"
```

## Fine-tuning SmolLM2 (Optional)

Fine-tuning is possible but slow on CPU (~200 sec/step). Use the base model for now.

```bash
# Prepare model and dataset
python prepare_chat.py

# LoRA fine-tune (WARNING: very slow on CPU, ~4 min/step)
python train_finetune.py --max-length 64 --grad-accum 8 --max-steps 300
```

## Results

### Fine-tuned LSTM (from scratch, 200 steps)
| Format | val_loss | Model Size | Compression |
|--------|----------|------------|-------------|
| FP32   | 7.0795   | 12.8 MB    | 1x          |
| INT4   | 6.5364   | 1.8 MB     | 7.2x        |
| INT2   | 6.6186   | 985 KB     | 13.0x       |

**Status**: Produces gibberish. Training iterations insufficient.

### SmolLM2-135M-Instruct (pre-trained, chat-ready)
| Model | Params | RAM | Status | Quality |
|-------|--------|-----|--------|---------|
| SmolLM2-135M-Instruct | 134.5M | ~400MB | **WORKS** | Good chat responses |

**Status**: ✅ Can chat! Multi-turn conversation supported. No fine-tuning needed.

## Model Architectures

### Fine-tuned (Legacy)
- **TinyLSTM**: Embedding → LSTM(d_model) → Linear → vocab
- **TinyTransformer**: Embedding + PosEnc → TransformerEncoder(2 layers) → Linear → vocab
- **INT4/INT2 variants**: Same architectures with STE quantization on all weights

### Pre-trained (Current)
- **SmolLM2-135M-Instruct**: Modern Llama-style architecture with SFT + DPO training
- Chat template: `<|im_start|>user/assistant`
- Supports multi-turn conversation with context

## Rules

- Models must be < 10M parameters (for from-scratch training)
- RAM must stay under 3.5GB
- No CUDA/GPU
- Dataset downloads < 500MB
- See `program.md` for full protocol

## Environment

Python 3.12 environment at `venv2/`:
```
torch 2.10.0+cpu
transformers 5.3.0
peft 0.18.1
accelerate 1.13.0
datasets 4.8.3
```

## Documentation

- `report.tex` — LaTeX document with full project documentation
- `spec/SPEC.md` — Detailed specification
- `spec/ARCHITECTURE.md` — Architecture design notes
- `spec/HANDOFF.md` — Session handoff notes
- `program.md` — Experiment protocol (Indonesian)
