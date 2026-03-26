#!/bin/bash
# Quick Start for SmLM C++ Engine on Termux/Android
# No PyTorch required - pure C++ inference

set -e

echo "======================================"
echo "SmLM C++ Engine - Quick Start"
echo "======================================"
echo ""

# Check if we're in Termux
if [[ "$PREFIX" == "/data/data/com.termux/files/usr" ]]; then
    echo "✓ Running in Termux"
else
    echo "⚠ Not running in Termux - some features may differ"
fi

# Step 1: Build engine
echo ""
echo "Step 1: Building C++ engine..."
cd engine_cpp
make clean > /dev/null 2>&1 || true
make
echo "✓ Engine built successfully"

# Step 2: Generate test model
echo ""
echo "Step 2: Generating test model..."
cd ..
python scripts/generate_test_model.py --output test_model.slm
echo "✓ Test model generated"

# Step 3: Run inference
echo ""
echo "Step 3: Running inference test..."
cd engine_cpp
./engine --model ../test_model.slm --input 0,1,2 --max_tokens 20
echo "✓ Inference test passed"

# Step 4: Benchmark
echo ""
echo "Step 4: Running benchmark..."
./engine --model ../test_model.slm --benchmark --num_runs 50
echo "✓ Benchmark complete"

echo ""
echo "======================================"
echo "All tests passed!"
echo "======================================"
echo ""
echo "Next steps:"
echo "  - Edit test_model.slm parameters in scripts/generate_test_model.py"
echo "  - Export real models with scripts/export_to_cpp.py (requires PyTorch)"
echo "  - Integrate with your own C++ code"
echo ""
echo "Usage examples:"
echo "  ./engine --model model.slm --input 0,1,2 --max_tokens 100"
echo "  ./engine --model model.slm --benchmark"
echo "  ./tokenizer encode hello world"
echo ""
