#!/usr/bin/env python3
"""Smoke test for repository structure - works without PyTorch."""

import sys
import json
from pathlib import Path


def test_imports():
    """Test that modules can be imported."""
    
    print("Testing imports...")
    
    try:
        from compressor.student import StudentConfig, StudentTransformer
        print("  ✓ compressor.student")
    except Exception as e:
        print(f"  ✗ compressor.student: {e}")
        return False
    
    try:
        from compressor.distill import DistillationConfig, DistillationTrainer
        print("  ✓ compressor.distill")
    except Exception as e:
        print(f"  ✗ compressor.distill: {e}")
        return False
    
    try:
        from compressor.export import export_to_onnx
        print("  ✓ compressor.export")
    except Exception as e:
        print(f"  ✗ compressor.export: {e}")
        return False
    
    try:
        from compressor.quantize import quantize_dynamic_int8
        print("  ✓ compressor.quantize")
    except Exception as e:
        print(f"  ✗ compressor.quantize: {e}")
        return False
    
    try:
        from benchmarks.runner import BenchmarkRunner, BenchmarkResult
        print("  ✓ benchmarks.runner")
    except Exception as e:
        print(f"  ✗ benchmarks.runner: {e}")
        return False
    
    try:
        from worker.queue import Worker, JobConfig
        print("  ✓ worker.queue")
    except Exception as e:
        print(f"  ✗ worker.queue: {e}")
        return False
    
    try:
        from eval.quality import EvalResult
        print("  ✓ eval.quality")
    except Exception as e:
        print(f"  ✗ eval.quality: {e}")
        return False
    
    return True


def test_config_loading():
    """Test that configs can be loaded."""
    
    print("\nTesting config loading...")
    
    configs = ['configs/smollm2-60m.yaml', 'configs/smollm2-100m.yaml']
    
    for config_path in configs:
        if not Path(config_path).exists():
            print(f"  ✗ {config_path}: not found")
            return False
        
        try:
            from compressor.distill import DistillationConfig
            config = DistillationConfig.from_yaml(config_path)
            print(f"  ✓ {config_path}")
        except Exception as e:
            print(f"  ✗ {config_path}: {e}")
            return False
    
    return True


def test_student_creation():
    """Test student model creation."""
    
    print("\nTesting student model creation...")
    
    try:
        from compressor.student import StudentConfig, StudentTransformer
        
        # Create a minimal config
        config = StudentConfig(
            vocab_size=1000,
            num_hidden_layers=2,
            num_attention_heads=4,
            hidden_size=128,
            intermediate_size=512,
        )
        
        # Create model
        model = StudentTransformer(config)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Created student with {num_params / 1e6:.3f}M parameters")
        
        # Test forward pass
        import torch
        input_ids = torch.randint(0, 1000, (1, 32))
        with torch.no_grad():
            output = model(input_ids)
        
        print(f"  ✓ Forward pass: {list(output.shape)}")
        
        return True
        
    except ImportError as e:
        print(f"  ⊘ Skipped (requires PyTorch): {e}")
        return None
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_numpy_inference():
    """Test NumPy inference (no PyTorch required)."""
    
    print("\nTesting NumPy inference...")
    
    try:
        import numpy as np
        from compressor.numpy_inference import NumPyTransformer, gelu, softmax
        
        # Create minimal config
        config = {
            'vocab_size': 1000,
            'hidden_size': 64,
            'num_hidden_layers': 2,
            'num_attention_heads': 4,
            'max_position_embeddings': 128,
        }
        
        # Create random weights for testing
        weights = {
            'embed_tokens': np.random.randn(1000, 64) * 0.02,
            'embed_pos': np.random.randn(128, 64) * 0.02,
            'ln_f.weight': np.ones(64),
            'ln_f.bias': np.zeros(64),
            'lm_head.weight': np.random.randn(1000, 64) * 0.02,
        }
        
        # Add block weights
        for i in range(2):
            weights[f'layers.{i}.attn.in_proj_weight'] = np.random.randn(64, 64*3) * 0.02
            weights[f'layers.{i}.attn.in_proj_bias'] = np.zeros(64*3)
            weights[f'layers.{i}.attn.out_proj.weight'] = np.random.randn(64, 64) * 0.02
            weights[f'layers.{i}.attn.out_proj.bias'] = np.zeros(64)
            weights[f'layers.{i}.mlp.0.weight'] = np.random.randn(64, 256) * 0.02
            weights[f'layers.{i}.mlp.0.bias'] = np.zeros(256)
            weights[f'layers.{i}.mlp.2.weight'] = np.random.randn(256, 64) * 0.02
            weights[f'layers.{i}.mlp.2.bias'] = np.zeros(64)
            weights[f'layers.{i}.ln_1.weight'] = np.ones(64)
            weights[f'layers.{i}.ln_1.bias'] = np.zeros(64)
            weights[f'layers.{i}.ln_2.weight'] = np.ones(64)
            weights[f'layers.{i}.ln_2.bias'] = np.zeros(64)
        
        # Create model
        model = NumPyTransformer(config, weights)
        print(f"  ✓ Created NumPy transformer")
        
        # Test forward pass
        input_ids = np.array([[1, 2, 3, 4, 5]])
        output = model.forward(input_ids)
        print(f"  ✓ Forward pass: {list(output.shape)}")
        
        # Test activation functions
        test_input = np.array([[-1, 0, 1, 2]])
        gelu_out = gelu(test_input)
        softmax_out = softmax(test_input)
        print(f"  ✓ Activation functions work")
        
        return True
        
    except ImportError as e:
        print(f"  ⊘ Skipped (requires NumPy): {e}")
        return None
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def test_directory_structure():
    """Test directory structure."""
    
    print("\nTesting directory structure...")
    
    required_dirs = [
        'compressor',
        'engine_cpp',
        'worker',
        'benchmarks',
        'eval',
        'configs',
        'scripts',
    ]
    
    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            print(f"  ✗ Missing directory: {dir_name}")
            return False
        print(f"  ✓ {dir_name}/")
    
    return True


def main():
    """Run all smoke tests."""
    
    print("=" * 50)
    print("SmLM Repository Smoke Test")
    print("=" * 50)
    
    results = []
    
    # Test directory structure
    results.append(('Directory structure', test_directory_structure()))
    
    # Test imports
    results.append(('Module imports', test_imports()))
    
    # Test config loading
    results.append(('Config loading', test_config_loading()))
    
    # Test NumPy inference (no PyTorch needed)
    results.append(('NumPy inference', test_numpy_inference()))
    
    # Test student creation (requires PyTorch)
    results.append(('Student creation', test_student_creation()))
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    passed = sum(1 for _, r in results if r is True)
    skipped = sum(1 for _, r in results if r is None)
    failed = sum(1 for _, r in results if r is False)
    
    print(f"Passed: {passed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\n❌ Some tests failed")
        return 1
    else:
        print("\n✓ All tests passed (or skipped)")
        return 0


if __name__ == '__main__':
    sys.exit(main())
