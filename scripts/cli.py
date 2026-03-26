#!/usr/bin/env python3
"""Command-line interface for SmLM model compressor."""

import argparse
import sys
from pathlib import Path


def cmd_train(args):
    """Run training."""
    from scripts.train import main as train_main
    sys.argv = ['train.py', '--config', args.config]
    if args.smoke_test:
        sys.argv.append('--smoke-test')
    if args.resume:
        sys.argv.extend(['--resume', args.resume])
    train_main()


def cmd_export(args):
    """Export model."""
    from scripts.export import main as export_main
    sys.argv = ['export.py', '--checkpoint', args.checkpoint]
    if args.output_dir:
        sys.argv.extend(['--output-dir', args.output_dir])
    if args.quantize:
        sys.argv.append('--quantize')
    if args.validate:
        sys.argv.append('--validate')
    if args.optimize:
        sys.argv.append('--optimize')
    export_main()


def cmd_benchmark(args):
    """Run benchmarks."""
    from scripts.benchmark import main as benchmark_main
    sys.argv = ['benchmark.py']
    if args.teacher:
        sys.argv.extend(['--teacher', args.teacher])
    if args.student:
        sys.argv.extend(['--student', args.student])
    if args.onnx:
        sys.argv.extend(['--onnx', args.onnx])
    if args.tag:
        sys.argv.extend(['--tag', args.tag])
    if args.output:
        sys.argv.extend(['--output', args.output])
    benchmark_main()


def cmd_results(args):
    """View results."""
    from scripts.view_results import main as results_main
    results_main()


def cmd_quick_start(args):
    """Show quick start guide."""
    from scripts.quick_start import GUIDE
    print(GUIDE)


def cmd_smoke_test(args):
    """Run smoke tests."""
    from scripts.smoke_test import main as smoke_test_main
    sys.exit(smoke_test_main())


def cmd_synthetic_data(args):
    """Generate synthetic data."""
    from scripts.synthetic_data import generate_synthetic_dataset
    generate_synthetic_dataset(
        vocab_size=args.vocab_size,
        num_samples=args.num_samples,
        max_seq_len=args.max_seq_len,
        output_path=args.output,
    )


def main():
    parser = argparse.ArgumentParser(
        prog='smlm',
        description='SmLM - Small Language Model Compressor',
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # train
    p_train = subparsers.add_parser('train', help='Run distillation training')
    p_train.add_argument('--config', required=True, help='Config YAML file')
    p_train.add_argument('--smoke-test', action='store_true', help='Quick smoke test')
    p_train.add_argument('--resume', help='Resume from checkpoint')
    p_train.set_defaults(func=cmd_train)
    
    # export
    p_export = subparsers.add_parser('export', help='Export model to ONNX')
    p_export.add_argument('--checkpoint', required=True, help='Model checkpoint')
    p_export.add_argument('--output-dir', default='artifacts', help='Output directory')
    p_export.add_argument('--quantize', action='store_true', help='Apply INT8 quantization')
    p_export.add_argument('--validate', action='store_true', help='Validate export')
    p_export.add_argument('--optimize', action='store_true', help='Optimize ONNX graph')
    p_export.set_defaults(func=cmd_export)
    
    # benchmark
    p_benchmark = subparsers.add_parser('benchmark', help='Run benchmarks')
    p_benchmark.add_argument('--teacher', help='Teacher model name/path')
    p_benchmark.add_argument('--student', help='Student checkpoint path')
    p_benchmark.add_argument('--onnx', help='ONNX model path')
    p_benchmark.add_argument('--tag', required=True, help='Experiment tag')
    p_benchmark.add_argument('--output', default='results/benchmarks.json', help='Output file')
    p_benchmark.set_defaults(func=cmd_benchmark)
    
    # results
    p_results = subparsers.add_parser('results', help='View experiment results')
    p_results.add_argument('--file', default='results.tsv', help='Results file')
    p_results.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    p_results.add_argument('--compare', nargs='+', help='Compare specific tags')
    p_results.add_argument('--export', help='Export to JSON')
    p_results.add_argument('--status', choices=['keep', 'discard', 'crash', 'all'], default='all')
    p_results.set_defaults(func=cmd_results)
    
    # quick-start
    p_quick = subparsers.add_parser('quick-start', help='Show quick start guide')
    p_quick.set_defaults(func=cmd_quick_start)
    
    # smoke-test
    p_smoke = subparsers.add_parser('smoke-test', help='Run smoke tests')
    p_smoke.set_defaults(func=cmd_smoke_test)
    
    # synthetic-data
    p_data = subparsers.add_parser('synthetic-data', help='Generate synthetic dataset')
    p_data.add_argument('--vocab-size', type=int, default=50257, help='Vocabulary size')
    p_data.add_argument('--num-samples', type=int, default=1000, help='Number of samples')
    p_data.add_argument('--max-seq-len', type=int, default=512, help='Max sequence length')
    p_data.add_argument('--output', default='data/synthetic_train.pt', help='Output path')
    p_data.set_defaults(func=cmd_synthetic_data)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == '__main__':
    main()
