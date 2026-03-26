# SmLM C++ CPU Inference Engine

Minimal C++ engine for running compressed transformer models.

## Build

```bash
cd engine_cpp
make
```

## Usage

```bash
# Run with exported model
./engine --model ../artifacts/model.onnx --prompt "Hello, world!" --max_tokens 100

# Benchmark
./engine --model ../artifacts/model.onnx --benchmark --num_runs 100
```

## Features

- CPU-only inference
- ONNX model loading
- Greedy and sampling generation
- Throughput and latency benchmarking
- Low memory footprint
