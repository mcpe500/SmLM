#!/bin/bash
# SmLM Setup Script

set -e

echo "=== SmLM Setup ==="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

# Create directories
echo "Creating directories..."
mkdir -p checkpoints artifacts results reports

# Install dependencies (optional - may fail on some platforms)
echo ""
echo "Installing dependencies..."
echo "Note: On Android/Termux, PyTorch may not be available."
echo "      The codebase can still be used for design and ONNX inference."
echo ""

pip3 install numpy tqdm pyyaml psutil 2>/dev/null || echo "Basic deps install skipped/failed"

echo ""
echo "Attempting to install ML dependencies (may fail on some platforms)..."
pip3 install torch transformers onnx onnxruntime datasets 2>/dev/null || {
    echo ""
    echo "ML dependencies installation failed."
    echo "This is expected on Android/Termux without PyTorch support."
    echo ""
    echo "You can still:"
    echo "  - Use the codebase for reference and design"
    echo "  - Run ONNX inference with pre-exported models"
    echo "  - Build the C++ engine: cd engine_cpp && make"
    echo ""
}

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Verify syntax: python3 -m py_compile scripts/*.py"
echo "  2. Build C++ engine: cd engine_cpp && make"
echo "  3. On a PyTorch-enabled machine:"
echo "     - Run training: python scripts/train.py --config configs/smollm2-60m.yaml"
echo "     - Export: python scripts/export.py --checkpoint ..."
echo "     - Benchmark: python scripts/benchmark.py --onnx ..."
