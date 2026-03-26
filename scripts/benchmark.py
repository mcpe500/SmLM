#!/usr/bin/env python3
"""Benchmark script for model comparison."""

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.runner import BenchmarkRunner, BenchmarkResult, save_benchmark_results, compare_results
from compressor.student import StudentTransformer, StudentConfig


def load_pytorch_model(model_path: str, is_student: bool = False):
    """Load PyTorch model."""
    
    if is_student:
        checkpoint = torch.load(model_path, map_location='cpu')
        config = checkpoint.get('student_config')
        model = StudentTransformer(config)
        model.load_state_dict(checkpoint.get('student_state_dict', checkpoint))
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return model, tokenizer
    
    return model, None


def main():
    parser = argparse.ArgumentParser(description='Benchmark models')
    parser.add_argument('--teacher', type=str, default='HuggingFaceTB/SmolLM2-360M',
                        help='Teacher model name or path')
    parser.add_argument('--student', type=str, default=None,
                        help='Student checkpoint path')
    parser.add_argument('--onnx', type=str, default=None,
                        help='ONNX model path')
    parser.add_argument('--seq-len', type=int, default=64, help='Sequence length')
    parser.add_argument('--num-runs', type=int, default=10, help='Number of runs')
    parser.add_argument('--output', type=str, default='results/benchmarks.json',
                        help='Output path for results')
    parser.add_argument('--tag', type=str, default='benchmark', help='Run tag')
    args = parser.parse_args()
    
    results = []
    
    # Benchmark teacher
    print(f"\n=== Benchmarking Teacher: {args.teacher} ===")
    teacher_model, _ = load_pytorch_model(args.teacher)
    teacher_runner = BenchmarkRunner(
        teacher_model,
        engine='pytorch',
        seq_len=args.seq_len,
        num_runs=args.num_runs,
    )
    teacher_result = teacher_runner.benchmark(args.teacher)
    results.append(teacher_result)
    
    # Benchmark student (PyTorch)
    if args.student:
        print(f"\n=== Benchmarking Student (PyTorch): {args.student} ===")
        student_model, _ = load_pytorch_model(args.student, is_student=True)
        student_runner = BenchmarkRunner(
            student_model,
            engine='pytorch',
            seq_len=args.seq_len,
            num_runs=args.num_runs,
        )
        student_result = student_runner.benchmark(args.student)
        results.append(student_result)
        
        # Compare
        comparison = compare_results(teacher_result, student_result)
        print(f"\nStudent vs Teacher comparison:")
        print(f"  - Size ratio: {comparison['size_ratio']:.2f}")
        print(f"  - Latency ratio: {comparison['latency_ratio']:.2f}")
        print(f"  - Throughput improvement: {comparison['throughput_improvement']*100:.1f}%")
    
    # Benchmark ONNX
    if args.onnx:
        print(f"\n=== Benchmarking ONNX: {args.onnx} ===")
        import onnxruntime as ort
        onnx_session = ort.InferenceSession(args.onnx)
        onnx_runner = BenchmarkRunner(
            onnx_session,
            engine='onnx',
            seq_len=args.seq_len,
            num_runs=args.num_runs,
        )
        onnx_result = onnx_runner.benchmark(args.onnx, artifact_path=args.onnx)
        results.append(onnx_result)
    
    # Save results
    save_benchmark_results(results, args.output)
    
    # Also append to results.tsv
    tsv_path = Path('results.tsv')
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    
    header = "commit\ttag\tlane\tmodel\tquality\tload_s\ttoks_per_s\tpeak_ram_gb\tartifact_mb\tstatus\tdescription"
    
    if not tsv_path.exists():
        with open(tsv_path, 'w') as f:
            f.write(header + '\n')
    
    with open(tsv_path, 'a') as f:
        for result in results:
            row = result.to_tsv_row(
                tag=args.tag,
                status='keep',
                description=f"Benchmark: {result.engine} engine",
            )
            f.write(row + '\n')
    
    print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
