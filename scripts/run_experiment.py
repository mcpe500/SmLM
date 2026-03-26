#!/usr/bin/env python3
"""Automated experiment runner - executes full compression pipeline."""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


@dataclass
class ExperimentConfig:
    """Configuration for an automated experiment run."""
    
    tag: str
    train_config: str
    teacher: str = "HuggingFaceTB/SmolLM2-360M"
    epochs: int = 10
    smoke_test: bool = False
    
    # Export options
    do_export: bool = True
    do_quantize: bool = True
    do_optimize: bool = True
    
    # Benchmark options
    do_benchmark: bool = True
    seq_len: int = 64
    num_runs: int = 10
    
    # Output
    output_dir: str = "results"
    artifacts_dir: str = "artifacts"
    checkpoints_dir: str = "checkpoints"
    
    def to_dict(self) -> dict:
        return {
            'tag': self.tag,
            'train_config': self.train_config,
            'teacher': self.teacher,
            'epochs': self.epochs,
            'smoke_test': self.smoke_test,
            'do_export': self.do_export,
            'do_quantize': self.do_quantize,
            'do_optimize': self.do_optimize,
            'do_benchmark': self.do_benchmark,
            'seq_len': self.seq_len,
            'num_runs': self.num_runs,
        }


class ExperimentRunner:
    """Runs complete compression experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.start_time = datetime.now()
        self.status = "pending"
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []
        
        # Paths
        self.log_dir = Path("logs") / config.tag
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.report_path = Path("reports") / config.tag
        self.report_path.mkdir(parents=True, exist_ok=True)
        
        self.artifacts_path = Path(config.artifacts_dir) / config.tag
        self.checkpoints_path = Path(config.checkpoints_dir) / config.tag
    
    def log(self, message: str):
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        
        log_file = self.log_dir / "run.log"
        with open(log_file, 'a') as f:
            f.write(line + '\n')
    
    def run_command(self, cmd: List[str], log_file: str = None) -> bool:
        """Run a shell command."""
        
        self.log(f"Running: {' '.join(cmd)}")
        
        if log_file:
            with open(log_file, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        else:
            result = subprocess.run(cmd)
        
        if result.returncode != 0:
            error = f"Command failed with exit code {result.returncode}"
            self.log(f"❌ {error}")
            self.errors.append(error)
            return False
        
        self.log("✓ Command completed successfully")
        return True
    
    def stage_train(self) -> bool:
        """Stage 1: Training."""
        
        self.log("\n" + "="*60)
        self.log("STAGE 1: Training")
        self.log("="*60)
        
        cmd = [
            sys.executable, "scripts/train.py",
            "--config", self.config.train_config,
        ]
        
        if self.config.smoke_test:
            cmd.append("--smoke-test")
            self.log("Running in smoke test mode")
        
        log_file = str(self.log_dir / "train.log")
        
        success = self.run_command(cmd, log_file)
        
        if success:
            # Find checkpoint
            final_ckpt = self.checkpoints_path / "final" / "model.pt"
            if not final_ckpt.exists():
                # Try smoke test path
                final_ckpt = self.checkpoints_path / f"{Path(self.config.train_config).stem}-smoke" / "final" / "model.pt"
            
            if final_ckpt.exists():
                self.results['checkpoint'] = str(final_ckpt)
                self.log(f"Checkpoint saved: {final_ckpt}")
            else:
                self.log("Warning: Could not find final checkpoint")
                self.results['checkpoint'] = str(self.checkpoints_path)
        
        return success
    
    def stage_export(self) -> bool:
        """Stage 2: Export to ONNX."""
        
        self.log("\n" + "="*60)
        self.log("STAGE 2: Export to ONNX")
        self.log("="*60)
        
        checkpoint = self.results.get('checkpoint')
        if not checkpoint:
            self.log("❌ No checkpoint found, skipping export")
            return False
        
        cmd = [
            sys.executable, "scripts/export.py",
            "--checkpoint", checkpoint,
            "--output-dir", str(self.artifacts_path),
        ]
        
        if self.config.do_validate:
            cmd.append("--validate")
        if self.config.do_optimize:
            cmd.append("--optimize")
        
        log_file = str(self.log_dir / "export.log")
        success = self.run_command(cmd, log_file)
        
        if success:
            onnx_path = self.artifacts_path / "model.onnx"
            if onnx_path.exists():
                self.results['onnx'] = str(onnx_path)
                size_mb = onnx_path.stat().st_size / 1024 / 1024
                self.log(f"ONNX model: {size_mb:.2f} MB")
        
        return success
    
    def stage_quantize(self) -> bool:
        """Stage 3: INT8 Quantization."""
        
        self.log("\n" + "="*60)
        self.log("STAGE 3: INT8 Quantization")
        self.log("="*60)
        
        onnx_path = self.results.get('onnx')
        if not onnx_path:
            self.log("❌ No ONNX model found, skipping quantization")
            return False
        
        cmd = [
            sys.executable, "-c",
            f"from compressor.quantize import quantize_dynamic_int8; "
            f"quantize_dynamic_int8('{onnx_path}', '{self.artifacts_path}/model_int8.onnx')"
        ]
        
        log_file = str(self.log_dir / "quantize.log")
        success = self.run_command(cmd, log_file)
        
        if success:
            int8_path = self.artifacts_path / "model_int8.onnx"
            if int8_path.exists():
                self.results['onnx_int8'] = str(int8_path)
                size_mb = int8_path.stat().st_size / 1024 / 1024
                self.log(f"INT8 model: {size_mb:.2f} MB")
        
        return success
    
    def stage_benchmark(self) -> bool:
        """Stage 4: Benchmarking."""
        
        self.log("\n" + "="*60)
        self.log("STAGE 4: Benchmarking")
        self.log("="*60)
        
        cmd = [
            sys.executable, "scripts/benchmark.py",
            "--teacher", self.config.teacher,
            "--tag", self.config.tag,
            "--seq-len", str(self.config.seq_len),
            "--num-runs", str(self.config.num_runs),
            "--output", str(self.report_path / "benchmarks.json"),
        ]
        
        if self.results.get('checkpoint'):
            cmd.extend(["--student", self.results['checkpoint']])
        if self.results.get('onnx_int8'):
            cmd.extend(["--onnx", self.results['onnx_int8']])
        
        log_file = str(self.log_dir / "benchmark.log")
        success = self.run_command(cmd, log_file)
        
        return success
    
    def generate_report(self):
        """Generate experiment summary report."""
        
        self.log("\n" + "="*60)
        self.log("Generating Report")
        self.log("="*60)
        
        end_time = datetime.now()
        duration = end_time - self.start_time
        
        report = {
            'tag': self.config.tag,
            'status': self.status,
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'config': self.config.to_dict(),
            'results': self.results,
            'errors': self.errors,
        }
        
        # Save JSON report
        report_path = self.report_path / "experiment.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.log(f"Report saved: {report_path}")
        
        # Save markdown summary
        md_path = self.report_path / "summary.md"
        with open(md_path, 'w') as f:
            f.write(f"# Experiment Summary: {self.config.tag}\n\n")
            f.write(f"**Status:** {self.status}\n\n")
            f.write(f"**Duration:** {duration.total_seconds()/60:.1f} minutes\n\n")
            
            f.write("## Configuration\n\n")
            for key, value in self.config.to_dict().items():
                f.write(f"- **{key}:** {value}\n")
            
            f.write("\n## Results\n\n")
            for key, value in self.results.items():
                f.write(f"- **{key}:** {value}\n")
            
            if self.errors:
                f.write("\n## Errors\n\n")
                for error in self.errors:
                    f.write(f"- {error}\n")
        
        self.log(f"Summary saved: {md_path}")
        
        # Append to results.tsv
        self._append_to_results_tsv()
    
    def _append_to_results_tsv(self):
        """Append result to results.tsv."""
        
        tsv_path = Path("results.tsv")
        header = "commit\ttag\tlane\tmodel\tquality\tload_s\ttoks_per_s\tpeak_ram_gb\tartifact_mb\tstatus\tdescription"
        
        # Read existing or create new
        if tsv_path.exists():
            with open(tsv_path, 'r') as f:
                existing = f.read().strip()
            if not existing:
                with open(tsv_path, 'w') as f:
                    f.write(header + '\n')
        else:
            with open(tsv_path, 'w') as f:
                f.write(header + '\n')
        
        # Get git commit
        try:
            commit = subprocess.check_output(
                ['git', 'rev-parse', '--short', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode().strip()
        except:
            commit = "workspace"
        
        # Determine status
        status = "keep" if self.status == "completed" else "crash"
        
        # Create row (placeholder values - real values from benchmark)
        row = [
            commit,
            self.config.tag,
            "long" if not self.config.smoke_test else "quick",
            Path(self.config.train_config).stem,
            "0.0",  # quality (from benchmark)
            "0.0",  # load_s
            "0.0",  # toks_per_s
            "0.0",  # peak_ram_gb
            "0.0",  # artifact_mb
            status,
            f"Auto-run: {'smoke test' if self.config.smoke_test else 'full pipeline'}"
        ]
        
        with open(tsv_path, 'a') as f:
            f.write('\t'.join(row) + '\n')
        
        self.log(f"Results appended to {tsv_path}")
    
    def run(self) -> bool:
        """Run complete experiment pipeline."""
        
        self.log(f"\n{'='*60}")
        self.log(f"Starting Experiment: {self.config.tag}")
        self.log(f"{'='*60}")
        self.log(f"Start time: {self.start_time}")
        
        self.status = "running"
        
        try:
            # Stage 1: Training (always run)
            if not self.stage_train():
                self.status = "failed_train"
                self.generate_report()
                return False
            
            # Stage 2: Export (optional)
            if self.config.do_export:
                if not self.stage_export():
                    self.status = "failed_export"
                    self.generate_report()
                    return False
            
            # Stage 3: Quantization (optional)
            if self.config.do_quantize:
                if not self.stage_quantize():
                    self.status = "failed_quantize"
                    self.generate_report()
                    return False
            
            # Stage 4: Benchmark (optional)
            if self.config.do_benchmark:
                if not self.stage_benchmark():
                    self.status = "failed_benchmark"
                    self.generate_report()
                    return False
            
            self.status = "completed"
            self.log("\n" + "="*60)
            self.log("✅ Experiment completed successfully!")
            self.log("="*60)
            
        except Exception as e:
            self.status = f"crashed: {str(e)}"
            self.log(f"\n❌ Experiment crashed: {e}")
            self.errors.append(str(e))
        
        finally:
            self.generate_report()
        
        return self.status == "completed"


def run_experiment(
    tag: str,
    config: str,
    smoke: bool = False,
    **kwargs
) -> bool:
    """Run a single experiment."""
    
    exp_config = ExperimentConfig(
        tag=tag,
        train_config=config,
        smoke_test=smoke,
        **kwargs
    )
    
    runner = ExperimentRunner(exp_config)
    return runner.run()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run automated experiment')
    parser.add_argument('--tag', required=True, help='Experiment tag')
    parser.add_argument('--config', required=True, help='Training config YAML')
    parser.add_argument('--teacher', default='HuggingFaceTB/SmolLM2-360M',
                        help='Teacher model')
    parser.add_argument('--smoke', action='store_true', help='Smoke test mode')
    parser.add_argument('--no-export', action='store_true', help='Skip export')
    parser.add_argument('--no-quantize', action='store_true', help='Skip quantization')
    parser.add_argument('--no-benchmark', action='store_true', help='Skip benchmark')
    
    args = parser.parse_args()
    
    success = run_experiment(
        tag=args.tag,
        config=args.config,
        teacher=args.teacher,
        smoke=args.smoke,
        do_export=not args.no_export,
        do_quantize=not args.no_quantize,
        do_benchmark=not args.no_benchmark,
    )
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
