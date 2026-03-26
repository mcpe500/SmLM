# SmLM C++ Inference Engine

Pure C++ transformer inference engine - **no external ML libraries required**.

## Quick Start (Termux/Android)

```bash
# 1. Build the engine
cd engine_cpp
make

# 2. Generate a test model (requires Python, no PyTorch needed)
python ../scripts/generate_test_model.py --output test.slm

# 3. Run inference
./engine --model test.slm --input 0,1,2 --max_tokens 20
```

## File Format

Models are saved in `.slm` (SmLM Model) binary format:

```
[magic: 4B] [version: 4B] [config: 24B] [weights...]
```

Weights are stored as: `[size: uint32] [data: float32[size]]`

## Building

### Termux/Android

```bash
cd engine_cpp
make
```

### Linux (x86_64)

```bash
cd engine_cpp
make
```

### macOS (Apple Silicon)

```bash
cd engine_cpp
make
```

### Debug Build

```bash
make DEBUG=1
```

## Usage

### Basic Inference

```bash
./engine --model model.slm --input 50256,123,456 --max_tokens 50
```

### Sampling

```bash
# Temperature sampling
./engine --model model.slm --input 0,1,2 --temperature 0.8 --top_k 40

# Greedy decoding (temperature=0)
./engine --model model.slm --input 0,1,2 --temperature 0
```

### Benchmark

```bash
./engine --model model.slm --benchmark --num_runs 100
```

### Help

```bash
./engine --help
```

## Exporting Models

### From HuggingFace (requires PyTorch)

```bash
# Export SmolLM2-135M
python scripts/export_to_cpp.py --model HuggingFaceTB/SmolLM2-135M --output smollm2-135m.slm

# Export GPT-2
python scripts/export_to_cpp.py --model gpt2 --output gpt2-small.slm
```

### From Checkpoint

```bash
python scripts/export_to_cpp.py --checkpoint checkpoints/model.pt --output model.slm
```

### Generate Test Model (No PyTorch)

```bash
python scripts/generate_test_model.py --output test.slm \
    --vocab-size 1000 \
    --hidden-size 128 \
    --layers 2 \
    --heads 4
```

## Tokenizer

Simple tokenizer for testing:

```bash
# Build tokenizer
make tokenizer

# Encode text
./tokenizer encode hello world model

# Decode IDs
./tokenizer decode 400,401,402
```

**Note:** For production use, integrate with a real BPE tokenizer (e.g., load tokenizer.json from HuggingFace).

## Model Configurations

### Test Model
- Vocab: 1000
- Hidden: 128
- Layers: 2
- Heads: 4
- Size: ~5 MB

### SmolLM2-30M (Student)
- Vocab: 50257
- Hidden: 384
- Layers: 4
- Heads: 4
- FFN: 1536
- Size: ~30 MB

### SmolLM2-60M (Student)
- Vocab: 50257
- Hidden: 512
- Layers: 6
- Heads: 6
- FFN: 2048
- Size: ~60 MB

### SmolLM2-135M (Teacher)
- Vocab: 50257
- Hidden: 768
- Layers: 12
- Heads: 12
- FFN: 3072
- Size: ~135 MB

## Memory Usage

Approximate RAM usage during inference:

| Model | RAM Usage |
|-------|-----------|
| Test (2L) | ~10 MB |
| 30M (4L) | ~100 MB |
| 60M (6L) | ~200 MB |
| 135M (12L) | ~500 MB |

## Performance

Expected tokens/sec on different hardware:

| Hardware | 30M | 60M | 135M |
|----------|-----|-----|------|
| Termux (Snapdragon) | ~50 t/s | ~20 t/s | ~5 t/s |
| Linux (8-core) | ~100 t/s | ~50 t/s | ~20 t/s |
| macOS (M1) | ~200 t/s | ~100 t/s | ~50 t/s |

## Architecture

The engine implements a standard decoder-only transformer:

1. **Token Embeddings** - Lookup table for input tokens
2. **Position Embeddings** - Learned positional encodings
3. **Transformer Layers** (N x):
   - LayerNorm
   - Self-Attention (Q, K, V projections + output)
   - LayerNorm
   - MLP (up + down projections with GELU)
4. **Final LayerNorm**
5. **LM Head** - Project to vocab logits

## Limitations

- **No KV cache** - Currently recomputes all positions each step (slower but simpler)
- **No batched inference** - Single sequence only
- **Basic tokenizer** - Uses simple word lookup, not BPE
- **FP32 only** - No INT8/FP16 quantization yet

## Future Improvements

- [ ] KV cache for faster generation
- [ ] BPE tokenizer integration
- [ ] INT8 quantization support
- [ ] Batched inference
- [ ] ONNX Runtime backend
- [ ] ARM NEON optimizations

## Troubleshooting

### Build fails on Termux

```bash
# Install build tools
pkg install clang make

# Then build
make
```

### Model load fails

Check the model file format:

```bash
# Check file header
xxd model.slm | head -n 5

# Should show: 53 6d 4c 4d (SmLM magic)
```

### Out of memory

Use a smaller model or reduce max_positions in config.

## License

Apache 2.0 - Same as main SmLM project
