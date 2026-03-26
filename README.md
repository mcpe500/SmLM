# SmLM - Small Language Model Compressor

Autonomous model compression research system for CPU deployment.

## Mission

Transform teacher models (e.g., SmolLM2-360M) into smaller, faster, lower-memory student models with a C++ CPU inference engine.

## Core Strategy

1. **Dense student shrinkage** - fewer layers, narrower width, smaller FFN
2. **Distillation** - knowledge transfer from teacher
3. **Quantization** - INT8 post-training quantization
4. **CPU engine** - C++ inference runtime

## Directory Structure

- `compressor/` - compression, distillation, pruning, export logic
- `engine_cpp/` - C++ CPU inference engine
- `worker/` - long-running job orchestration
- `benchmarks/` - latency, memory, artifact comparisons
- `eval/` - quality evaluation harness
- `configs/` - experiment configs
- `scripts/` - operational scripts
- `checkpoints/` - model checkpoints
- `artifacts/` - exported models
- `results/` - benchmarks and reports

## Requirements

### Full Training/Export (requires PyTorch)

```bash
pip install torch transformers onnx onnxruntime datasets tqdm pyyaml psutil
```

### Inference/Benchmark Only (ONNX Runtime)

```bash
pip install onnx onnxruntime numpy tqdm pyyaml psutil
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build C++ engine
cd engine_cpp && make

# Run baseline teacher benchmark (requires PyTorch)
python scripts/benchmark.py --model smollm2-360m --output results/teacher_baseline.tsv

# Run distillation (requires PyTorch)
python scripts/train.py --config configs/smollm2-60m.yaml

# Export and quantize (requires PyTorch checkpoint)
python scripts/export.py --checkpoint checkpoints/model.pt --output artifacts/model.onnx
python -c "from compressor.quantize import quantize_dynamic_int8; quantize_dynamic_int8('artifacts/model.onnx', 'artifacts/model_int8.onnx')"

# Benchmark compressed model (ONNX Runtime only)
python scripts/benchmark.py --onnx artifacts/model_int8.onnx --tag my-experiment
```

## Experiment Loop

1. Create branch: `git checkout -b autoresearch/<tag>`
2. Run experiment in appropriate lane (quick/long/deploy/repair)
3. Log results to `results.tsv`
4. Keep/discard based on metrics

## Hardware Assumptions

- CPU only
- Limited RAM (target <4GB peak)
- 24/7 worker for long jobs
- Wall-clock time secondary to robustness

## Success Criteria

1. Quality retention (within acceptable drop)
2. Inference speed improvement
3. Memory reduction
4. Deployability (exports and runs)
