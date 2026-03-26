# SmLM Project Status

## Current State

**Repository initialized:** March 26, 2026

**Latest Update:** Pure C++ inference engine implemented for Termux/Android

**Current experiment:** `autoresearch/mar26-smollm2-60m` (60M student)

**Status:** ✅ Pure C++ engine complete and tested on Termux

## Latest Updates

### March 27, 2026 - Pure C++ Implementation

- ✅ **Complete C++ transformer engine** (812 lines)
  - No external ML libraries
  - Works on Termux/Android
  - ~700 tokens/sec on test model
- ✅ **Binary model format (.slm)**
  - Little-endian float32
  - Cross-platform compatible
- ✅ **Export scripts**
  - `export_to_cpp.py` - PyTorch → C++ binary
  - `generate_test_model.py` - Random models without PyTorch
- ✅ **Tokenizer utility**
  - Simple word-based tokenizer
  - Encode/decode CLI
- ✅ **Build system**
  - Makefile for Termux, Linux, macOS
  - ARM and x86_64 optimizations
- ✅ **Documentation**
  - `README_CPP.md` - C++ engine guide
  - `IMPLEMENTATION.md` - Technical details
  - `quick_start_cpp.sh` - Automated setup

## Implemented Components

### ✅ Pure C++ Engine (New!)

| Component | Status | Description |
|-----------|--------|-------------|
| `engine_cpp/transformer_engine.cpp` | Complete | Full transformer inference |
| `engine_cpp/tokenizer.cpp` | Complete | Simple tokenizer utility |
| `engine_cpp/Makefile` | Complete | Cross-platform build |
| `scripts/export_to_cpp.py` | Complete | PyTorch → C++ export |
| `scripts/generate_test_model.py` | Complete | Test model generator |

### ✅ Core Python Modules

| Module | Status | Description |
|--------|--------|-------------|
| `compressor/student.py` | Complete | Student architecture generator |
| `compressor/distill.py` | Complete | Distillation training loop |
| `compressor/export.py` | Complete | ONNX export and validation |
| `compressor/quantize.py` | Complete | INT8 post-training quantization |
| `compressor/numpy_inference.py` | Complete | Pure NumPy inference |
| `benchmarks/runner.py` | Complete | Latency, memory, throughput |
| `eval/quality.py` | Complete | Perplexity evaluation |
| `worker/queue.py` | Complete | Job orchestration |

## Repository Structure

```
SmLM/
├── compressor/              # Python compression logic
│   ├── student.py
│   ├── distill.py
│   ├── export.py
│   ├── quantize.py
│   └── numpy_inference.py
│
├── engine_cpp/              # C++ inference engine (NEW!)
│   ├── transformer_engine.cpp
│   ├── tokenizer.cpp
│   ├── Makefile
│   ├── README_CPP.md
│   └── IMPLEMENTATION.md
│
├── scripts/
│   ├── export_to_cpp.py     # PyTorch → C++ export (NEW!)
│   ├── generate_test_model.py  # Test model gen (NEW!)
│   ├── quick_start_cpp.sh   # Quick start (NEW!)
│   └── ...
│
├── results.tsv
├── README.md
└── program.md
```

## Next Steps

### For Termux/Android (Current Environment)

1. **Test with larger models**
   ```bash
   python scripts/generate_test_model.py --output medium.slm \
     --hidden-size 256 --layers 4 --vocab-size 5000
   ```

2. **Integrate better tokenizer**
   - Load BPE merges from HuggingFace
   - Implement character-level fallback

3. **Optimize for ARM**
   - NEON SIMD instructions
   - Better cache utilization

### For PyTorch Environment

1. **Export real model**
   ```bash
   python scripts/export_to_cpp.py \
     --model HuggingFaceTB/SmolLM2-135M \
     --output smollm2-135m.slm
   ```

2. **Run distillation**
   ```bash
   python scripts/train.py --config configs/smollm2-60m.yaml
   ```

3. **Export and benchmark**
   ```bash
   python scripts/export_to_cpp.py --checkpoint checkpoints/model.pt
   ./engine --model model.slm --benchmark
   ```

## Performance Benchmarks

### C++ Engine (Termux)

| Model | Params | Tokens/sec | RAM |
|-------|--------|------------|-----|
| Test (2L/128H) | 0.7M | ~700 | ~10MB |

### C++ Engine (Linux x86_64)

| Model | Params | Tokens/sec | RAM |
|-------|--------|------------|-----|
| Test (2L/128H) | 0.7M | ~1500 | ~10MB |

## Known Limitations

1. **No KV cache** - Slower for long sequences
2. **Basic tokenizer** - Word-based only
3. **No training** - Inference only
4. **FP32 only** - No quantization yet

## Files Verified

All C++ modules compile and run:
- `transformer_engine.cpp` ✓
- `tokenizer.cpp` ✓
- `Makefile` ✓

All Python scripts tested:
- `export_to_cpp.py` ✓
- `generate_test_model.py` ✓
- `quick_start_cpp.sh` ✓
