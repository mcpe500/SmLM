# Model Architecture Specification

## Architecture Candidates

### A. TinyTransformer (Priority 1)
```
Embedding(vocab_size, d_model)
2x TransformerDecoderLayer(d_model, nhead=2, dim_feedforward=d_model*2)
Linear(d_model, vocab_size)
```
- Params estimate: ~2-5M with d_model=128, vocab=50257
- Pros: Modern, good quality/param ratio
- Cons: Attention has O(n^2) memory

### B. TinyLSTM (Priority 2)
```
Embedding(vocab_size, d_model)
2-layer LSTM(d_model, d_model, num_layers=2, batch_first=True)
Linear(d_model, vocab_size)
```
- Params estimate: ~1-3M with d_model=64-128
- Pros: O(n) memory, proven architecture
- Cons: Slower training, less quality per param

### C. TinyGRU (Priority 3)
```
Embedding(vocab_size, d_model)
2-layer GRU(d_model, d_model, num_layers=2, batch_first=True)
Linear(d_model, vocab_size)
```
- Params estimate: ~1-2M with d_model=64-128
- Pros: Faster than LSTM, O(n) memory
- Cons: Slightly less capacity than LSTM

## Baseline Configuration (Run 1)
- Architecture: TinyLSTM
- d_model: 64
- num_layers: 2
- vocab_size: 256 (character-level) or 50257 (GPT-2 tokenizer)
- Estimated params: ~500K - 2M

## Scaling Plan
1. Run 1: Character-level LSTM, ~500K params (proving ground)
2. Run 2: Character-level LSTM, ~2M params
3. Run 3: BPE tokenizer + LSTM, ~3M params
4. Run 4: TinyTransformer, 2 layers, ~3M params
5. Run 5+: Iterate on best architecture
