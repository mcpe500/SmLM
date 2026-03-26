#!/usr/bin/env python3
"""View and analyze experiment results from results.tsv."""

import argparse
from pathlib import Path
from typing import List, Dict, Any
import json


def load_results(path: str = "results.tsv") -> List[Dict[str, Any]]:
    """Load results from TSV file."""
    
    path = Path(path)
    if not path.exists():
        print(f"Results file not found: {path}")
        return []
    
    results = []
    with open(path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        return []
    
    # Parse header
    header = lines[0].strip().split('\t')
    
    # Parse data rows
    for line in lines[1:]:
        if not line.strip():
            continue
        
        values = line.strip().split('\t')
        if len(values) != len(header):
            continue
        
        row = dict(zip(header, values))
        
        # Convert numeric fields
        for field in ['quality', 'load_s', 'toks_per_s', 'peak_ram_gb', 'artifact_mb']:
            try:
                row[field] = float(row[field]) if row[field] != '0.0' else 0.0
            except (ValueError, KeyError):
                pass
        
        results.append(row)
    
    return results


def format_results(results: List[Dict[str, Any]], verbose: bool = False):
    """Format results for display."""
    
    if not results:
        print("No results found.")
        return
    
    print(f"\n{'='*80}")
    print(f"Experiment Results ({len(results)} runs)")
    print(f"{'='*80}\n")
    
    # Group by tag
    by_tag = {}
    for r in results:
        tag = r.get('tag', 'unknown')
        if tag not in by_tag:
            by_tag[tag] = []
        by_tag[tag].append(r)
    
    for tag, runs in sorted(by_tag.items()):
        print(f"\n### {tag}")
        print(f"{'-'*60}")
        
        for run in runs:
            status_icon = {
                'keep': '✅',
                'discard': '❌',
                'crash': '💥',
            }.get(run.get('status', ''), '❓')
            
            print(f"\n{status_icon} [{run.get('status', '?')}] {run.get('lane', '?')} - {run.get('model', '?')}")
            print(f"   Commit: {run.get('commit', '?')}")
            
            # Metrics
            quality = run.get('quality', 0)
            if isinstance(quality, float) and quality > 0:
                print(f"   Quality: {quality:.4f}")
            
            toks_per_s = run.get('toks_per_s', 0)
            if isinstance(toks_per_s, float) and toks_per_s > 0:
                print(f"   Throughput: {toks_per_s:.1f} toks/s")
            
            peak_ram = run.get('peak_ram_gb', 0)
            if isinstance(peak_ram, float) and peak_ram > 0:
                print(f"   Peak RAM: {peak_ram:.2f} GB")
            
            artifact_mb = run.get('artifact_mb', 0)
            if isinstance(artifact_mb, float) and artifact_mb > 0:
                print(f"   Artifact: {artifact_mb:.1f} MB")
            
            # Description
            if verbose and run.get('description'):
                print(f"   Note: {run.get('description')}")
        
        # Summary for tag
        if len(runs) > 1:
            keeps = sum(1 for r in runs if r.get('status') == 'keep')
            print(f"\n   Summary: {keeps}/{len(runs)} runs kept")


def compare_tags(results: List[Dict[str, Any]], tags: List[str]):
    """Compare specific tags side by side."""
    
    print(f"\n{'='*80}")
    print(f"Comparison: {', '.join(tags)}")
    print(f"{'='*80}\n")
    
    filtered = [r for r in results if r.get('tag') in tags]
    
    if not filtered:
        print("No matching results found.")
        return
    
    # Find best run for each tag
    best_by_tag = {}
    for tag in tags:
        tag_runs = [r for r in filtered if r.get('tag') == tag and r.get('status') == 'keep']
        if tag_runs:
            # Sort by quality (lower is better for perplexity)
            best = min(tag_runs, key=lambda x: x.get('quality', float('inf')))
            best_by_tag[tag] = best
    
    if not best_by_tag:
        print("No 'keep' status runs found for comparison.")
        return
    
    # Print comparison table
    print(f"{'Metric':<20}", end='')
    for tag in tags:
        print(f"{tag:<20}", end='')
    print()
    
    print(f"{'-'*20}" * (1 + len(tags)))
    
    metrics = [
        ('quality', 'Quality'),
        ('toks_per_s', 'Toks/s'),
        ('peak_ram_gb', 'RAM (GB)'),
        ('artifact_mb', 'Size (MB)'),
    ]
    
    for field, label in metrics:
        print(f"{label:<20}", end='')
        for tag in tags:
            if tag in best_by_tag:
                val = best_by_tag[tag].get(field, 0)
                if isinstance(val, float):
                    print(f"{val:<20.4f}", end='')
                else:
                    print(f"{str(val):<20}", end='')
            else:
                print(f"{'-':<20}", end='')
        print()


def export_json(results: List[Dict[str, Any]], output_path: str):
    """Export results to JSON."""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Exported {len(results)} results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='View experiment results')
    parser.add_argument('--file', type=str, default='results.tsv',
                        help='Path to results.tsv file')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed descriptions')
    parser.add_argument('--compare', type=str, nargs='+',
                        help='Compare specific tags')
    parser.add_argument('--export', type=str,
                        help='Export results to JSON file')
    parser.add_argument('--status', type=str,
                        choices=['keep', 'discard', 'crash', 'all'],
                        default='all', help='Filter by status')
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.file)
    
    if not results:
        print("No results to display.")
        return
    
    # Filter by status
    if args.status != 'all':
        results = [r for r in results if r.get('status') == args.status]
    
    # Compare specific tags
    if args.compare:
        compare_tags(results, args.compare)
    else:
        format_results(results, verbose=args.verbose)
    
    # Export if requested
    if args.export:
        export_json(results, args.export)


if __name__ == '__main__':
    main()
