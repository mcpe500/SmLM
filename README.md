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
inference.py        Interactive chat / text generation
int4_train_test.py  Pure INT4 training with STE (7.2x compression)
int2_train_test.py  Pure INT2 training with STE (13.0x compression)
results.tsv         Experiment log
report.tex          LaTeX report (project documentation)
pyproject.toml      Dependencies
program.md          Experiment protocol and rules
spec/
  SPEC.md           Detailed specification
  ARCHITECTURE.md   Architecture candidates
```

## Setup

```bash
# Use conda/mamba environment with Python 3.12
export PATH="$HOME/.local/share/mamba/envs/slm/bin:$PATH"

# Or install manually
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets numpy accelerate
```

## Quick Start

```bash
# 1. Prepare data
python prepare.py

# 2. Train (baseline FP32 LSTM)
python train.py --arch lstm --d-model 32 --num-layers 1 --epochs 2 --max-steps 200

# 3. Chat with trained model
python inference.py --arch lstm --d-model 32 --num-layers 1 --prompt "Hello"
```

## Quantization Experiments

Pure INT4 and INT2 training using Straight-Through Estimator (STE). Weights stored and updated at reduced precision, gradients flow through unchanged.

```bash
# INT4 training (7.2x smaller than FP32)
python int4_train_test.py

# INT2 training (13.0x smaller than FP32)
python int2_train_test.py
```

## Results

| Format | val_loss | Model Size | Compression | Commit |
|--------|----------|------------|-------------|--------|
| FP32 | 7.0795 | 12.8 MB | 1x | `9ffbf9f` |
| INT4 | 6.5364 | 1.8 MB | 7.2x | — |
| INT2 | 6.6186 | 985 KB | 13.0x | — |

All models trained: 200 steps × 2 epochs, LSTM 1-layer, d_model=32, 3.28M params, TinyStories subset.

**Current status**: Models produce gibberish. 400 total training steps is insufficient. The architecture and pipeline are sound — the bottleneck is training iterations, not memory or precision.

## Model Architectures

- **TinyLSTM**: Embedding → LSTM(d_model) → Linear → vocab
- **TinyTransformer**: Embedding + PosEnc → TransformerEncoder(2 layers) → Linear → vocab
- **INT4/INT2 variants**: Same architectures with STE quantization on all weights

## Rules

- Models must be < 10M parameters
- RAM must stay under 3.5GB
- No CUDA/GPU
- Dataset downloads < 500MB
- See `program.md` for full protocol

## Documentation

- `report.tex` — LaTeX document with full project documentation
- `spec/SPEC.md` — Detailed specification
- `spec/ARCHITECTURE.md` — Architecture design notes
- `program.md` — Experiment protocol (Indonesian)
