"""Worker orchestration for long-running jobs."""

import os
import json
import time
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any, List


@dataclass
class JobConfig:
    """Configuration for a worker job."""
    
    job_id: str
    job_type: str  # train, export, benchmark
    command: str
    args: Dict[str, Any]
    priority: int = 0
    timeout_hours: float = 24.0
    resume_on_crash: bool = True
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JobStatus:
    """Status of a worker job."""
    
    job_id: str
    status: str  # pending, running, completed, failed, crashed
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    progress: float = 0.0
    current_step: str = ""
    error: Optional[str] = None
    checkpoint_path: Optional[str] = None
    log_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        return asdict(self)


class Worker:
    """Worker for long-running compression jobs."""
    
    def __init__(self, work_dir: str = "worker"):
        self.work_dir = Path(work_dir)
        self.jobs_dir = self.work_dir / "jobs"
        self.status_dir = self.work_dir / "status"
        self.logs_dir = self.work_dir / "logs"
        
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.status_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_job: Optional[JobConfig] = None
        self.running = False
    
    def submit_job(self, config: JobConfig) -> str:
        """Submit a job to the worker queue."""
        
        job_path = self.jobs_dir / f"{config.job_id}.json"
        
        with open(job_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Initialize status
        status = JobStatus(job_id=config.job_id, status='pending')
        self._update_status(status)
        
        print(f"Submitted job: {config.job_id}")
        return config.job_id
    
    def _update_status(self, status: JobStatus):
        """Update job status file."""
        
        status_path = self.status_dir / f"{status.job_id}.json"
        
        with open(status_path, 'w') as f:
            json.dump(status.to_dict(), f, indent=2)
    
    def run_job(self, config: JobConfig) -> JobStatus:
        """Run a single job."""
        
        self.current_job = config
        self.running = True
        
        status = JobStatus(
            job_id=config.job_id,
            status='running',
            start_time=datetime.now().isoformat(),
            current_step='starting',
        )
        self._update_status(status)
        
        # Prepare command
        cmd = config.command.split()
        for key, value in config.args.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            else:
                cmd.append(f"--{key}")
                cmd.append(str(value))
        
        # Setup log file
        log_path = self.logs_dir / f"{config.job_id}.log"
        
        print(f"Running job: {config.job_id}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Log: {log_path}")
        
        try:
            # Run with timeout
            timeout_seconds = int(config.timeout_hours * 3600)
            
            with open(log_path, 'w') as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                )
                
                # Monitor progress
                start_time = time.time()
                
                while process.poll() is None:
                    elapsed = time.time() - start_time
                    
                    # Check timeout
                    if elapsed > timeout_seconds:
                        process.kill()
                        status.status = 'crashed'
                        status.error = f"Timeout after {elapsed/3600:.2f} hours"
                        self._update_status(status)
                        return status
                    
                    # Update progress (simplified - would parse logs in reality)
                    status.progress = min(0.99, elapsed / timeout_seconds)
                    self._update_status(status)
                    
                    time.sleep(10)
                
                # Check exit code
                if process.returncode == 0:
                    status.status = 'completed'
                    status.progress = 1.0
                    status.current_step = 'done'
                else:
                    status.status = 'failed'
                    status.error = f"Exit code: {process.returncode}"
                
                status.end_time = datetime.now().isoformat()
                status.log_path = str(log_path)
                self._update_status(status)
                
                print(f"Job {config.job_id} {status.status}")
                return status
                
        except Exception as e:
            status.status = 'crashed'
            status.error = str(e)
            status.end_time = datetime.now().isoformat()
            self._update_status(status)
            
            print(f"Job {config.job_id} crashed: {e}")
            return status
        
        finally:
            self.running = False
            self.current_job = None
    
    def get_status(self, job_id: str) -> Optional[JobStatus]:
        """Get job status."""
        
        status_path = self.status_dir / f"{job_id}.json"
        
        if not status_path.exists():
            return None
        
        with open(status_path, 'r') as f:
            data = json.load(f)
        
        return JobStatus(**data)
    
    def list_jobs(self) -> List[str]:
        """List all job IDs."""
        
        return [p.stem for p in self.status_dir.glob("*.json")]
    
    def run_queue(self):
        """Run jobs from the queue."""
        
        print("Starting worker queue processor...")
        
        while True:
            # Find pending jobs
            pending = []
            for job_path in self.jobs_dir.glob("*.json"):
                job_id = job_path.stem
                status = self.get_status(job_id)
                if status and status.status == 'pending':
                    with open(job_path, 'r') as f:
                        config_data = json.load(f)
                    config = JobConfig(**config_data)
                    pending.append(config)
            
            if pending:
                # Sort by priority
                pending.sort(key=lambda x: -x.priority)
                job = pending[0]
                
                # Run job
                self.run_job(job)
                
                # Remove completed job from queue
                if self.get_status(job.job_id).status in ['completed', 'failed', 'crashed']:
                    (self.jobs_dir / f"{job.job_id}.json").unlink()
            else:
                time.sleep(60)  # Wait for new jobs


def create_train_job(
    job_id: str,
    config_path: str,
    resume: bool = False,
    timeout_hours: float = 24.0,
) -> JobConfig:
    """Create a training job."""
    
    return JobConfig(
        job_id=job_id,
        job_type='train',
        command='python scripts/train.py',
        args={
            'config': config_path,
            'resume': resume,
        },
        timeout_hours=timeout_hours,
        resume_on_crash=True,
    )


def create_export_job(
    job_id: str,
    checkpoint_path: str,
    quantize: bool = True,
    timeout_hours: float = 2.0,
) -> JobConfig:
    """Create an export job."""
    
    return JobConfig(
        job_id=job_id,
        job_type='export',
        command='python scripts/export.py',
        args={
            'checkpoint': checkpoint_path,
            'quantize': quantize,
            'validate': True,
        },
        timeout_hours=timeout_hours,
    )


def create_benchmark_job(
    job_id: str,
    student_path: Optional[str] = None,
    onnx_path: Optional[str] = None,
    timeout_hours: float = 1.0,
) -> JobConfig:
    """Create a benchmark job."""
    
    args = {'tag': job_id}
    
    if student_path:
        args['student'] = student_path
    if onnx_path:
        args['onnx'] = onnx_path
    
    return JobConfig(
        job_id=job_id,
        job_type='benchmark',
        command='python scripts/benchmark.py',
        args=args,
        timeout_hours=timeout_hours,
    )
