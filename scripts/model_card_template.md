---
license: apache-2.0
tags:
  - compression
  - distillation
  - student-model
  - cpu-optimized
---

# {{ model_name }}

{{ model_description }}

## Model Details

- **Teacher:** {{ teacher_name }} ({{ teacher_params }}M parameters)
- **Student:** {{ student_name }} ({{ student_params }}M parameters)
- **Compression Ratio:** {{ compression_ratio }}x
- **Method:** Knowledge Distillation
- **Dataset:** {{ dataset_name }}

## Architecture

| Component | Teacher | Student | Ratio |
|-----------|---------|---------|-------|
| Layers | {{ teacher_layers }} | {{ student_layers }} | {{ layer_ratio }} |
| Hidden Size | {{ teacher_hidden }} | {{ student_hidden }} | {{ hidden_ratio }} |
| Attention Heads | {{ teacher_heads }} | {{ student_heads }} | {{ heads_ratio }} |
| FFN Size | {{ teacher_ffn }} | {{ student_ffn }} | {{ ffn_ratio }} |

## Training

- **Distillation Temperature:** {{ temperature }}
- **Learning Rate:** {{ learning_rate }}
- **Batch Size:** {{ batch_size }}
- **Epochs:** {{ num_epochs }}
- **Training Time:** {{ training_time }}

## Performance

### Quality Metrics

| Metric | Teacher | Student | Retention |
|--------|---------|---------|-----------|
| Perplexity | {{ teacher_perplexity }} | {{ student_perplexity }} | {{ quality_retention }}% |
| Loss | {{ teacher_loss }} | {{ student_loss }} | - |

### Runtime Metrics (CPU)

| Metric | Teacher | Student | Improvement |
|--------|---------|---------|-------------|
| Load Time | {{ teacher_load_time }}s | {{ student_load_time }}s | {{ load_improvement }}x |
| Throughput | {{ teacher_toks_per_s }} toks/s | {{ student_toks_per_s }} toks/s | {{ throughput_improvement }}x |
| Peak RAM | {{ teacher_ram }}GB | {{ student_ram }}GB | {{ memory_improvement }}x |
| Model Size | {{ teacher_size }}MB | {{ student_size }}MB | {{ size_improvement }}x |

## Quantized Version

The INT8 quantized version provides additional benefits:

| Metric | FP32 | INT8 | Improvement |
|--------|------|------|-------------|
| Size | {{ fp32_size }}MB | {{ int8_size }}MB | {{ int8_size_reduction }}% |
| Throughput | {{ fp32_toks_per_s }} toks/s | {{ int8_toks_per_s }} toks/s | {{ int8_speedup }}x |

## Usage

### PyTorch

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("{{ repo_id }}")
# ... use model
```

### ONNX Runtime

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {'input': input_ids})
```

### C++ Engine

```bash
./engine --model model.onnx --prompt "Hello" --max_tokens 100
```

## Files

- `model.pt` - PyTorch checkpoint
- `model.onnx` - ONNX export (FP32)
- `model_int8.onnx` - ONNX export (INT8 quantized)
- `config.json` - Model configuration
- `tokenizer.json` - Tokenizer files

## Training Code

See the original repository: https://github.com/mcpe500/SmLM

```bash
# Train this model
python scripts/train.py --config configs/{{ config_name }}.yaml

# Export
python scripts/export.py --checkpoint checkpoints/{{ checkpoint_path }} --quantize

# Benchmark
python scripts/benchmark.py --student checkpoints/{{ checkpoint_path }} --tag {{ tag }}
```

## Limitations

- Quality may be lower than teacher on complex reasoning tasks
- Optimized for CPU inference; GPU performance not benchmarked
- Trained on wikitext-2; domain adaptation may be needed

## License

Apache License 2.0

## Citation

```bibtex
@software{smlm2026,
  title = {SmLM: Small Language Model Compressor},
  author = {SmLM Team},
  year = {2026},
  url = {https://github.com/mcpe500/SmLM}
}
```
