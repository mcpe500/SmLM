#!/usr/bin/env python3
"""Export script for compressed models."""

import argparse
import torch
from pathlib import Path

from compressor.student import StudentTransformer, StudentConfig
from compressor.export import export_to_onnx, validate_onnx_output, optimize_onnx
from compressor.quantize import quantize_dynamic_int8


def main():
    parser = argparse.ArgumentParser(description='Export compressed model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--output-dir', type=str, default='artifacts', help='Output directory')
    parser.add_argument('--seq-len', type=int, default=512, help='Sequence length for export')
    parser.add_argument('--quantize', action='store_true', help='Apply INT8 quantization')
    parser.add_argument('--optimize', action='store_true', help='Optimize ONNX graph')
    parser.add_argument('--validate', action='store_true', help='Validate export')
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    student_config = checkpoint.get('student_config')
    if student_config is None:
        # Try to infer from state dict
        state_dict = checkpoint.get('student_state_dict', checkpoint)
        # This is simplified - in practice you'd need the actual config
        student_config = StudentConfig()
    
    # Create model
    student = StudentTransformer(student_config)
    
    state_dict = checkpoint.get('student_state_dict', checkpoint)
    student.load_state_dict(state_dict)
    student.eval()
    
    print(f"Loaded model with {sum(p.numel() for p in student.parameters()) / 1e6:.2f}M parameters")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    print(f"Exporting to ONNX: {onnx_path}")
    
    export_to_onnx(
        student,
        student_config,
        str(onnx_path),
        seq_len=args.seq_len,
    )
    
    # Validate
    if args.validate:
        print("Validating ONNX export...")
        is_valid, max_diff = validate_onnx_output(student, str(onnx_path))
        print(f"Validation: {'PASS' if is_valid else 'FAIL'} (max_diff={max_diff:.6f})")
    
    # Optimize
    if args.optimize:
        optimized_path = output_dir / "model_optimized.onnx"
        print(f"Optimizing: {optimized_path}")
        optimize_onnx(str(onnx_path), str(optimized_path))
        onnx_path = optimized_path
    
    # Quantize
    if args.quantize:
        quantized_path = output_dir / "model_int8.onnx"
        print(f"Quantizing to INT8: {quantized_path}")
        quantize_dynamic_int8(str(onnx_path), str(quantized_path))
    
    print("Export complete!")


if __name__ == '__main__':
    main()
