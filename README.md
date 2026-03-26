# SmLM - Small Language Model Compressor

[![GitHub](https://img.shields.io/github/repo-size/mcpe500/SmLM)](https://github.com/mcpe500/SmLM)
[![License](https://img.shields.io/github/license/mcpe500/SmLM)](https://github.com/mcpe500/SmLM/blob/main/LICENSE)

Autonomous model compression research system for CPU deployment. Transform large language models into smaller, faster, CPU-optimized students.

## 🎯 Mission

Compress teacher models (e.g., **SmolLM2-360M**) into smaller students (**30-180M**) using:
1. **Dense shrinkage** - fewer layers, narrower width, smaller FFN
2. **Knowledge distillation** - transfer teacher knowledge
3. **INT8 quantization** - reduce size and improve CPU speed
4. **ONNX export** - deployment-ready format

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers onnx onnxruntime datasets tqdm pyyaml psutil
```

### 2. Run a Smoke Test (10 min)

```bash
# Minimal test with synthetic data
python scripts/run_experiment.py --tag test-1 --config configs/minimal-test.yaml --smoke
```

### 3. Full Compression Run (2-4 hours)

```bash
# Compress SmolLM2-360M → 60M student
python scripts/run_experiment.py --tag mar26-60m --config configs/smollm2-60m.yaml
```

### 4. View Results

```bash
# Show all results
python scripts/cli.py results

# Compare experiments
python scripts/cli.py results --compare mar26-60m mar26-100m
```

## 📊 Available Configurations

| Config | Student Size | Layers | Hidden | Heads | Use Case |
|--------|-------------|--------|--------|-------|----------|
| `minimal-test.yaml` | ~1M | 2 | 128 | 4 | Pipeline testing |
| `smollm2-30m.yaml` | ~30M | 4 | 384 | 4 | Maximum compression |
| `smollm2-60m.yaml` | ~60M | 6 | 512 | 6 | **Recommended baseline** |
| `smollm2-100m.yaml` | ~100M | 8 | 640 | 8 | Quality-focused |
| `smollm2-180m.yaml` | ~180M | 12 | 768 | 12 | Conservative compression |

## 📖 CLI Commands

```bash
# Training
python scripts/cli.py train --config configs/smollm2-60m.yaml
python scripts/cli.py train --config configs/smollm2-60m.yaml --smoke-test

# Export
python scripts/cli.py export --checkpoint model.pt --quantize --validate

# Benchmark
python scripts/cli.py benchmark --student model.pt --onnx model_int8.onnx --tag test

# Results
python scripts/cli.py results --verbose
python scripts/cli.py results --compare mar26-60m mar26-100m

# Utilities
python scripts/cli.py quick-start    # Show guide
python scripts/cli.py smoke-test     # Verify setup
python scripts/cli.py synthetic-data # Generate test data
```

## 🔄 Automated Pipeline

The `run_experiment.py` script automates the full compression pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│  Experiment: mar26-60m                                      │
├─────────────────────────────────────────────────────────────┤
│  Stage 1: Training                                          │
│    → Distillation from SmolLM2-360M                         │
│    → Checkpoint: checkpoints/mar26-60m/final/model.pt      │
├─────────────────────────────────────────────────────────────┤
│  Stage 2: Export                                            │
│    → ONNX: artifacts/mar26-60m/model.onnx                  │
│    → Validated against PyTorch output                       │
├─────────────────────────────────────────────────────────────┤
│  Stage 3: Quantization                                      │
│    → INT8: artifacts/mar26-60m/model_int8.onnx             │
│    → ~50% size reduction                                    │
├─────────────────────────────────────────────────────────────┤
│  Stage 4: Benchmark                                         │
│    → Teacher: perplexity, latency, memory                   │
│    → Student (PT): perplexity, latency, memory              │
│    → Student (ONNX INT8): latency, memory                   │
├─────────────────────────────────────────────────────────────┤
│  Output: reports/mar26-60m/                                 │
│    - experiment.json (metrics)                              │
│    - summary.md (human-readable)                            │
│    - benchmarks.json (detailed results)                     │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Repository Structure

```
SmLM/
├── compressor/              # Core compression logic
│   ├── student.py          # Student architecture generator
│   ├── distill.py          # Distillation training loop
│   ├── export.py           # ONNX export & validation
│   ├── quantize.py         # INT8 quantization
│   └── numpy_inference.py  # Pure NumPy inference (no PyTorch)
│
├── engine_cpp/              # C++ CPU inference engine
│   ├── engine.cpp          # Engine skeleton
│   └── Makefile
│
├── benchmarks/              # Performance benchmarking
│   └── runner.py           # Latency, throughput, memory
│
├── eval/                    # Quality evaluation
│   └── quality.py          # Perplexity, bits-per-byte
│
├── worker/                  # Job orchestration
│   └── queue.py            # Long-running job manager
│
├── configs/                 # Experiment configurations
│   ├── minimal-test.yaml   # 2-layer test config
│   ├── smollm2-30m.yaml    # 4-layer aggressive
│   ├── smollm2-60m.yaml    # 6-layer baseline
│   ├── smollm2-100m.yaml   # 8-layer quality
│   └── smollm2-180m.yaml   # 12-layer conservative
│
├── scripts/                 # CLI and utilities
│   ├── cli.py              # Unified CLI
│   ├── train.py            # Training script
│   ├── export.py           # Export script
│   ├── benchmark.py        # Benchmark script
│   ├── run_experiment.py   # Automated pipeline
│   ├── view_results.py     # Results viewer
│   ├── synthetic_data.py   # Data generator
│   └── quick_start.py      # Interactive guide
│
├── results.tsv              # Experiment tracking
└── reports/                 # Experiment reports
    └── <tag>/
        ├── experiment.json
        ├── summary.md
        └── benchmarks.json
```

## 📈 Success Criteria

A compression experiment is **successful** if:

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Size reduction** | >80% | 360M → 60M = 83% reduction |
| **Quality retention** | >70% | Perplexity ratio |
| **Throughput improvement** | >2x | Tokens/sec on CPU |
| **Memory reduction** | >50% | Peak RAM usage |

### Decision Policy

- **KEEP**: Meets targets without unacceptable regressions
- **DISCARD**: No practical improvement or quality collapse
- **CRASH**: Pipeline failure (export, benchmark, etc.)

## 🔬 Experiment Loop

Follow the autonomous agent protocol from `program.md`:

```bash
# 1. Create experiment branch
git checkout -b autoresearch/<tag>

# 2. Run experiment
python scripts/run_experiment.py --tag <tag> --config <config>

# 3. Review results
python scripts/cli.py results --compare <tag>

# 4. Commit and push
git add -A && git commit -m "Results: <tag>" && git push
```

## 🖥️ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8GB | 16GB |
| **Storage** | 10GB | 20GB |
| **CPU** | 4 cores | 8+ cores |
| **GPU** | Optional | CUDA-capable (speeds up training 5-10x) |

### CPU-Only Training

Expected training times for SmolLM2-60M (10 epochs on wikitext-2):
- **CPU (8 cores)**: ~4-6 hours
- **GPU (RTX 3060)**: ~30-45 minutes

## 📝 Example Results

```tsv
commit	tag	lane	model	quality	load_s	toks_per_s	peak_ram_gb	artifact_mb	status	description
abc123	mar26-60m	long	smollm2-60m	25.4	1.2	45.3	2.1	240.5	keep	6-layer baseline
def456	mar26-30m	long	smollm2-30m	28.9	0.9	62.1	1.5	125.2	keep	4-layer aggressive
ghi789	mar26-100m	long	smollm2-100m	23.1	1.5	38.7	2.8	380.1	discard	Diminishing returns
```

## 🛠️ Troubleshooting

### Out of Memory
```yaml
# In config YAML
training:
  batch_size: 8  # Reduce from 16
  gradient_accumulation_steps: 8  # Increase to compensate
```

### Training Too Slow
```bash
# Use GPU if available
# Or reduce dataset size
dataset:
  num_samples: 5000  # Reduce from 10000
```

### Export Fails
```bash
# Upgrade ONNX
pip install --upgrade onnx onnxruntime

# Try without optimization
python scripts/export.py --checkpoint model.pt --no-optimize
```

## 📚 Documentation

- [`README.md`](README.md) - This file
- [`program.md`](program.md) - Autonomous agent specification
- [`STATUS.md`](STATUS.md) - Current project status
- [`reports/<tag>/plan.md`](reports/) - Individual experiment plans
- [`scripts/quick_start.py`](scripts/quick_start.py) - Interactive guide (`python scripts/cli.py quick-start`)

## 🤝 Contributing

1. Fork the repository
2. Create experiment branch: `git checkout -b autoresearch/<tag>`
3. Run experiments following `program.md` protocol
4. Log results to `results.tsv`
5. Push and create pull request

## 📄 License

Apache License 2.0 - See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for transformers and datasets
- [ONNX Runtime](https://onnxruntime.ai/) for CPU optimization
- [SmolLM2](https://github.com/huggingface/smollm) for baseline models

---

**Current Status:** ✅ Codebase complete, ready for PyTorch-enabled environment

**Latest Experiment:** `autoresearch/mar26-smollm2-60m`

**Repository:** https://github.com/mcpe500/SmLM
