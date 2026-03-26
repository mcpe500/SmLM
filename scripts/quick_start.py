#!/usr/bin/env python3
"""Quick start guide for running experiments."""

GUIDE = """
# SmLM Quick Start Guide

## Prerequisites

You need a machine with:
- Python 3.10+
- PyTorch (CPU or CUDA)
- 8GB+ RAM (16GB recommended)
- 10GB+ disk space

## Setup (5 min)

```bash
# Clone repository
git clone https://github.com/mcpe500/SmLM.git
cd SmLM

# Install dependencies
pip install torch transformers onnx onnxruntime datasets tqdm pyyaml psutil

# Verify installation
python scripts/smoke_test.py
```

## Experiment 1: Smoke Test (10 min)

Run a minimal distillation to verify everything works:

```bash
# Generate synthetic data (no download needed)
python scripts/synthetic_data.py --num-samples 500

# Run smoke test distillation
python scripts/train.py --config configs/smollm2-60m.yaml --smoke-test
```

Expected output:
- Training starts within 30 seconds
- Loss decreases from ~10 to ~8
- Checkpoint saved to `checkpoints/smollm2-60m-v1-smoke/`

## Experiment 2: Full Distillation (2-4 hours)

Train a 60M student from SmolLM2-360M teacher:

```bash
python scripts/train.py --config configs/smollm2-60m.yaml
```

This will:
- Download wikitext-2 dataset (~7MB)
- Train for 10 epochs
- Save checkpoints every 5000 steps
- Take 2-4 hours on CPU, 30-60 min on GPU

Monitor progress:
```bash
tail -f run.log
```

## Experiment 3: Export and Quantize (5 min)

Export the trained student to ONNX and apply INT8 quantization:

```bash
# Export to ONNX
python scripts/export.py \\
  --checkpoint checkpoints/smollm2-60m-v1/final/model.pt \\
  --output-dir artifacts/mar26-smollm2-60m \\
  --validate --optimize

# Apply INT8 quantization
python -c "
from compressor.quantize import quantize_dynamic_int8
quantize_dynamic_int8('artifacts/mar26-smollm2-60m/model.onnx', 
                      'artifacts/mar26-smollm2-60m/model_int8.onnx')
"
```

Expected outputs:
- `model.onnx` (~240MB for 60M model)
- `model_int8.onnx` (~120MB, 50% smaller)

## Experiment 4: Benchmark (10 min)

Compare teacher vs student performance:

```bash
python scripts/benchmark.py \\
  --teacher HuggingFaceTB/SmolLM2-360M \\
  --student checkpoints/smollm2-60m-v1/final/model.pt \\
  --onnx artifacts/mar26-smollm2-60m/model_int8.onnx \\
  --tag mar26-smollm2-60m
```

Expected results:
- Student should be 2-4x faster than teacher
- INT8 should be slightly faster than FP32 ONNX
- Memory usage should be 50-70% lower

## Interpreting Results

Check `results.tsv` for metrics:

```bash
cat results.tsv
```

Key columns:
- `quality`: Lower is better (perplexity)
- `toks_per_s`: Higher is better (throughput)
- `peak_ram_gb`: Lower is better (memory)
- `artifact_mb`: Lower is better (size)
- `status`: keep/discard/crash

## Troubleshooting

### Out of memory
- Reduce batch_size in config
- Use gradient_accumulation_steps
- Reduce max_seq_len

### Training too slow
- Use GPU if available
- Reduce num_epochs for testing
- Use smaller dataset

### Export fails
- Check ONNX version: `pip install --upgrade onnx onnxruntime`
- Try without optimization: remove `--optimize` flag

### Quality too low
- Increase training epochs
- Lower distillation temperature (try 1.5)
- Use larger student (try smollm2-100m config)

## Next Steps

After successful baseline:

1. **Architecture sweep**: Try different layer/hidden ratios
2. **Temperature sweep**: Test KD temperatures 1.5, 2.0, 3.0
3. **Dataset sweep**: Try OpenWebText, C4 subsets
4. **Quantization comparison**: Dynamic vs static INT8

## Commands Reference

```bash
# Training
python scripts/train.py --config <config.yaml> [--smoke-test] [--resume <checkpoint>]

# Export
python scripts/export.py --checkpoint <model.pt> --output-dir <dir> [--quantize] [--validate]

# Benchmark
python scripts/benchmark.py --teacher <name> --student <path> --onnx <path> --tag <tag>

# Data
python scripts/synthetic_data.py --num-samples <n> --output <path>

# Smoke test
python scripts/smoke_test.py
```

## Configuration Files

- `configs/smollm2-60m.yaml`: 6 layers, 512 hidden, ~60M params
- `configs/smollm2-100m.yaml`: 8 layers, 640 hidden, ~100M params

Modify these to create new experiment configurations.

## Results Tracking

All experiments are logged to `results.tsv` in tab-separated format:

```
commit  tag  lane  model  quality  load_s  toks_per_s  peak_ram_gb  artifact_mb  status  description
```

Add new rows manually or use the benchmark script.

---

For detailed documentation, see:
- `README.md`: Project overview
- `program.md`: Agent specification
- `STATUS.md`: Current project status
- `reports/mar26-smollm2-60m/plan.md`: Current experiment plan
"""

if __name__ == '__main__':
    print(GUIDE)
