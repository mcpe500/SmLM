#!/usr/bin/env python3
"""
Log experiment result to results.tsv

Usage:
    python scripts/log_result.py --tag mar27-cpp-engine --lane quick --model test-model-2L --toks 533 --status keep --desc "Pure C++ engine test"
"""

import argparse
import subprocess
import os
from datetime import datetime


def get_git_commit():
    """Get short git commit hash or 'workspace' if uncommitted changes."""
    try:
        # Check if there are uncommitted changes
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
        )
        if result.stdout.strip():
            return 'workspace'
        
        # Get commit hash
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(__file__))
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    
    return 'workspace'


def log_result(tag, lane, model, toks, status, desc, 
               quality='N/A', load_s=0.0, peak_ram=0.0, artifact_mb=0.0):
    """Append result to results.tsv"""
    
    commit = get_git_commit()
    
    # TSV line
    line = f"{commit}\t{tag}\t{lane}\t{model}\t{quality}\t{load_s}\t{toks}\t{peak_ram}\t{artifact_mb}\t{status}\t{desc}\n"
    
    # Path to results.tsv
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    results_path = os.path.join(root_dir, 'results.tsv')
    
    # Append
    with open(results_path, 'a') as f:
        f.write(line)
    
    print(f"Logged result to {results_path}")
    print(f"  Tag: {tag}")
    print(f"  Model: {model}")
    print(f"  Status: {status}")
    print(f"  Tokens/sec: {toks}")
    print(f"  Description: {desc}")


def main():
    parser = argparse.ArgumentParser(description='Log experiment result to results.tsv')
    
    parser.add_argument('--tag', required=True, help='Experiment tag (e.g., mar27-cpp-engine)')
    parser.add_argument('--lane', required=True, choices=['quick', 'long', 'deploy', 'repair'],
                       help='Experiment lane')
    parser.add_argument('--model', required=True, help='Model identifier')
    parser.add_argument('--toks', type=float, required=True, help='Tokens per second')
    parser.add_argument('--status', required=True, choices=['keep', 'discard', 'crash'],
                       help='Experiment status')
    parser.add_argument('--desc', required=True, help='Description')
    parser.add_argument('--quality', default='N/A', help='Quality metric (default: N/A)')
    parser.add_argument('--load-s', type=float, default=0.0, help='Load time in seconds')
    parser.add_argument('--ram', type=float, default=0.0, help='Peak RAM in GB')
    parser.add_argument('--artifact-mb', type=float, default=0.0, help='Artifact size in MB')
    
    args = parser.parse_args()
    
    log_result(
        tag=args.tag,
        lane=args.lane,
        model=args.model,
        toks=args.toks,
        status=args.status,
        desc=args.desc,
        quality=args.quality,
        load_s=args.load_s,
        peak_ram=args.ram,
        artifact_mb=args.artifact_mb
    )


if __name__ == '__main__':
    main()
