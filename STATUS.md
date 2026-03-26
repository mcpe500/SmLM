# SmLM Project Status

## Current State

**Repository initialized:** March 26, 2026

**Target teacher:** SmolLM2-360M

**Status:** Codebase complete, awaiting PyTorch-enabled environment for training

## Implemented Components

### ✅ Core Modules

| Module | Status | Description |
|--------|--------|-------------|
| `compressor/student.py` | Complete | Student architecture generator with dense shrinkage |
| `compressor/distill.py` | Complete | Distillation training loop with checkpointing |
| `compressor/export.py` | Complete | ONNX export and validation |
| `compressor/quantize.py` | Complete | INT8 post-training quantization |
| `compressor/numpy_inference.py` | Complete | Pure NumPy inference (no PyTorch) |
| `benchmarks/runner.py` | Complete | Latency, memory, throughput benchmarking |
| `eval/quality.py` | Complete | Perplexity and quality evaluation |
| `worker/queue.py` | Complete | Long-running job orchestration |
| `engine_cpp/` | Complete | C++ CPU inference engine (skeleton) |

### ✅ Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train.py` | Distillation training |
| `scripts/export.py` | Model export to ONNX |
| `scripts/benchmark.py` | Model benchmarking |
| `scripts/smoke_test.py` | Repository verification |
| `scripts/setup.sh` | Environment setup |

### ✅ Configs

| Config | Student Size |
|--------|-------------|
| `configs/smollm2-60m.yaml` | ~60M (6 layers, 512 hidden) |
| `configs/smollm2-100m.yaml` | ~100M (8 layers, 640 hidden) |

## Repository Structure

```
SmLM/
├── compressor/
│   ├── __init__.py
│   ├── student.py          # Student architecture
│   ├── distill.py          # Distillation training
│   ├── export.py           # ONNX export
│   ├── quantize.py         # INT8 quantization
│   └── numpy_inference.py  # NumPy inference
├── engine_cpp/
│   ├── README.md
│   ├── Makefile
│   └── engine.cpp          # C++ engine skeleton
├── worker/
│   ├── __init__.py
│   └── queue.py            # Job orchestration
├── benchmarks/
│   ├── __init__.py
│   └── runner.py           # Benchmark harness
├── eval/
│   ├── __init__.py
│   └── quality.py          # Quality evaluation
├── configs/
│   ├── __init__.py
│   ├── smollm2-60m.yaml
│   └── smollm2-100m.yaml
├── scripts/
│   ├── __init__.py
│   ├── train.py
│   ├── export.py
│   ├── benchmark.py
│   ├── smoke_test.py
│   └── setup.sh
├── results.tsv             # Experiment results
├── requirements.txt
├── README.md
└── program.md              # Agent specification
```

## Next Steps

### Immediate (requires PyTorch environment)

1. **Install dependencies** on a machine with PyTorch support:
   ```bash
   pip install torch transformers onnx onnxruntime datasets
   ```

2. **Run smoke test** to verify setup:
   ```bash
   python scripts/smoke_test.py
   ```

3. **Run baseline benchmark** (teacher model):
   ```bash
   python scripts/benchmark.py --teacher HuggingFaceTB/SmolLM2-360M --output results/baseline.json
   ```

4. **Start distillation** (quick smoke test first):
   ```bash
   python scripts/train.py --config configs/smollm2-60m.yaml --smoke-test
   ```

5. **Export and quantize**:
   ```bash
   python scripts/export.py --checkpoint checkpoints/smollm2-60m-v1-smoke/final/model.pt --quantize --validate
   ```

6. **Benchmark compressed model**:
   ```bash
   python scripts/benchmark.py --onnx artifacts/model_int8.onnx --tag mar26-smollm2-60m
   ```

### For This Environment (Android/Termux)

Since PyTorch is not available:

1. **Build C++ engine** (if g++ is available):
   ```bash
   cd engine_cpp && make
   ```

2. **Use codebase as reference** for implementing on a PyTorch-enabled machine

3. **ONNX inference only** - if you have pre-exported ONNX models:
   ```bash
   pip install onnx onnxruntime
   python -c "import onnxruntime as ort; s = ort.InferenceSession('model.onnx'); print(s.get_inputs()[0])"
   ```

## Experiment Protocol

When running on a PyTorch-enabled machine:

1. **Create branch**: `git checkout -b autoresearch/<tag>`
2. **Run experiment** in appropriate lane (quick/long/deploy/repair)
3. **Log results** to `results.tsv`
4. **Evaluate** keep/discard based on metrics

### Success Criteria

- **Quality**: Student perplexity within acceptable range of teacher
- **Speed**: CPU latency improvement after export/quantization
- **Memory**: Reduced RAM footprint
- **Deployability**: Exports and runs in C++ engine

## Known Limitations

1. **No PyTorch on Android/Termux** - Training requires a different machine
2. **C++ engine is a skeleton** - Needs ONNX Runtime C API integration for full functionality
3. **No dataset caching** - Assumes network access for dataset download

## Files Verified

All Python modules pass syntax checking:
- `compressor/*.py` ✓
- `benchmarks/*.py` ✓
- `eval/*.py` ✓
- `worker/*.py` ✓
- `scripts/*.py` ✓
