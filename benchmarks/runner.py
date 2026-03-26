"""Benchmark harness for latency, memory, and throughput."""

import os
import json
import time
import psutil
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import torch


@dataclass
class BenchmarkResult:
    """Benchmark result."""
    
    model: str
    engine: str  # pytorch, onnx, cpp
    
    # Size metrics
    artifact_size_mb: float = 0.0
    
    # Time metrics
    load_time_s: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # Throughput
    tokens_per_sec: float = 0.0
    
    # Memory
    peak_memory_mb: float = 0.0
    avg_memory_mb: float = 0.0
    
    # Quality (optional)
    quality_score: Optional[float] = None
    
    # Metadata
    seq_len: int = 64
    num_runs: int = 10
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_tsv_row(self, tag: str, status: str, description: str) -> str:
        """Convert to TSV row for results.tsv."""
        return "\t".join([
            "workspace",
            tag,
            "deploy",
            self.model,
            str(self.quality_score or 0.0),
            f"{self.load_time_s:.3f}",
            f"{self.tokens_per_sec:.1f}",
            f"{self.peak_memory_mb / 1024:.2f}",
            f"{self.artifact_size_mb:.1f}",
            status,
            description,
        ])


class BenchmarkRunner:
    """Benchmark runner for model inference."""
    
    def __init__(
        self,
        model: Any,
        engine: str = "pytorch",
        device: str = "cpu",
        seq_len: int = 64,
        num_runs: int = 10,
        warmup_runs: int = 3,
    ):
        self.model = model
        self.engine = engine
        self.device = torch.device(device)
        self.seq_len = seq_len
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs
        
        if engine == "pytorch":
            self.model = self.model.to(self.device)
            self.model.eval()
    
    def _get_memory_mb(self) -> float:
        """Get current process memory in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _measure_latency(self, input_tensor: torch.Tensor) -> List[float]:
        """Measure inference latency."""
        
        latencies = []
        memory_samples = []
        
        # Warmup
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                if self.engine == "pytorch":
                    self.model(input_tensor)
        
        # Measure
        for _ in range(self.num_runs):
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            memory_samples.append(self._get_memory_mb())
            
            start = time.perf_counter()
            with torch.no_grad():
                if self.engine == "pytorch":
                    self.model(input_tensor)
            torch.cuda.synchronize() if self.device.type == "cuda" else None
            end = time.perf_counter()
            
            latencies.append((end - start) * 1000)  # ms
        
        return latencies, memory_samples
    
    def benchmark(self, model_name: str, artifact_path: Optional[str] = None) -> BenchmarkResult:
        """Run full benchmark."""
        
        print(f"Running benchmark for {model_name} ({self.engine})")
        
        # Create dummy input
        input_tensor = torch.randint(
            0, 1000,
            (1, self.seq_len),
            dtype=torch.long,
            device=self.device,
        )
        
        # Measure load time
        start = time.perf_counter()
        if artifact_path and self.engine == "onnx":
            import onnxruntime as ort
            self.model = ort.InferenceSession(artifact_path)
        load_time = time.perf_counter() - start
        
        # Measure latency
        if self.engine == "onnx":
            latencies = []
            memory_samples = []
            for _ in range(self.warmup_runs + self.num_runs):
                memory_samples.append(self._get_memory_mb())
                start = time.perf_counter()
                self.model.run(None, {'input': input_tensor.cpu().numpy()})
                end = time.perf_counter()
                latencies.append((end - start) * 1000)
            latencies = latencies[self.warmup_runs:]
            memory_samples = memory_samples[self.warmup_runs:]
        else:
            latencies, memory_samples = self._measure_latency(input_tensor)
        
        # Calculate metrics
        avg_latency = sum(latencies) / len(latencies)
        sorted_latencies = sorted(latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2]
        p95_idx = int(len(sorted_latencies) * 0.95)
        p99_idx = int(len(sorted_latencies) * 0.99)
        p95 = sorted_latencies[min(p95_idx, len(sorted_latencies) - 1)]
        p99 = sorted_latencies[min(p99_idx, len(sorted_latencies) - 1)]
        
        # Throughput (tokens per second)
        tokens_generated = self.seq_len * self.num_runs
        total_time = sum(latencies) / 1000
        tokens_per_sec = tokens_generated / total_time if total_time > 0 else 0
        
        # Memory
        peak_memory = max(memory_samples)
        avg_memory = sum(memory_samples) / len(memory_samples)
        
        # Artifact size
        artifact_size = 0.0
        if artifact_path and os.path.exists(artifact_path):
            artifact_size = os.path.getsize(artifact_path) / 1024 / 1024
        
        result = BenchmarkResult(
            model=model_name,
            engine=self.engine,
            artifact_size_mb=round(artifact_size, 2),
            load_time_s=round(load_time, 3),
            avg_latency_ms=round(avg_latency, 3),
            p50_latency_ms=round(p50, 3),
            p95_latency_ms=round(p95, 3),
            p99_latency_ms=round(p99, 3),
            tokens_per_sec=round(tokens_per_sec, 1),
            peak_memory_mb=round(peak_memory, 2),
            avg_memory_mb=round(avg_memory, 2),
            seq_len=self.seq_len,
            num_runs=self.num_runs,
        )
        
        print(f"Benchmark complete:")
        print(f"  - Load time: {result.load_time_s:.3f}s")
        print(f"  - Avg latency: {result.avg_latency_ms:.3f}ms")
        print(f"  - Tokens/sec: {result.tokens_per_sec:.1f}")
        print(f"  - Peak memory: {result.peak_memory_mb:.2f}MB")
        print(f"  - Artifact size: {result.artifact_size_mb:.2f}MB")
        
        return result


def save_benchmark_results(
    results: List[BenchmarkResult],
    output_path: str,
):
    """Save benchmark results to JSON."""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'results': [r.to_dict() for r in results],
        'summary': {
            'num_models': len(results),
            'engines': list(set(r.engine for r in results)),
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved benchmark results to {output_path}")


def compare_results(
    baseline: BenchmarkResult,
    candidate: BenchmarkResult,
) -> Dict[str, float]:
    """Compare two benchmark results."""
    
    def safe_ratio(a, b):
        return a / b if b != 0 else 0.0
    
    return {
        'size_ratio': safe_ratio(candidate.artifact_size_mb, baseline.artifact_size_mb),
        'size_improvement': 1 - safe_ratio(candidate.artifact_size_mb, baseline.artifact_size_mb),
        'latency_ratio': safe_ratio(candidate.avg_latency_ms, baseline.avg_latency_ms),
        'latency_improvement': 1 - safe_ratio(candidate.avg_latency_ms, baseline.avg_latency_ms),
        'throughput_ratio': safe_ratio(candidate.tokens_per_sec, baseline.tokens_per_sec),
        'throughput_improvement': safe_ratio(candidate.tokens_per_sec, baseline.tokens_per_sec) - 1,
        'memory_ratio': safe_ratio(candidate.peak_memory_mb, baseline.peak_memory_mb),
        'memory_improvement': 1 - safe_ratio(candidate.peak_memory_mb, baseline.peak_memory_mb),
    }
