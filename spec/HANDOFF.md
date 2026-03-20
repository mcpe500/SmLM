# Handoff Document - SLM CPU Experiment

## Session 1: Initial Bootstrap (2026-03-20)

### Environment Assessment
- **Python**: 3.13.7 (Termux)
- **pip**: 26.0.1
- **Available RAM**: ~1.4GB free of 7.1GB total (will constrain to 4GB budget)
- **CPU**: 8 cores (will use 1 per spec)
- **Disk**: 30GB free
- **Platform**: Android/Termux

### Key Decisions & Rationale
1. **PyTorch CPU**: Using `--index-url https://download.pytorch.org/whl/cpu` to avoid CUDA bloat
2. **Python 3.13**: Termux ships 3.13, may have compatibility issues with some packages - will test
3. **TinyStories dataset**: Small, simple language, good for tiny model baseline
4. **Custom tokenizer vs pre-trained**: Starting with character-level or small BPE for minimal overhead

### Pitfalls to Avoid
- [ ] DO NOT install CUDA version of PyTorch (wastes RAM/disk)
- [ ] DO NOT use datasets > 500MB
- [ ] DO NOT let model exceed 10M params
- [ ] DO NOT use mixed precision on CPU (unstable)
- [ ] DO NOT forget to set `OMP_NUM_THREADS=1` for single-core constraint
- [ ] DO NOT forget gradient accumulation for effective batch size without OOM

### What Works (updated as we iterate)
- TBD

### What Doesn't Work (updated as we iterate)
- TBD

---

## Experiment Log

### Run 1: Baseline (TBD)
- Status: Pending
- Architecture: TBD
- Result: TBD
