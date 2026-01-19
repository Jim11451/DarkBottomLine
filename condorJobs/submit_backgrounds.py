#!/usr/bin/env python3
"""
Simple script to submit condor jobs for background files.

For each background file (bkg_*.txt):
  - Counts the number of ROOT files in it
  - Submits N condor jobs (one job per file)
  - Each background gets its own condor cluster

Usage:
    python3 submit_backgrounds.py [options]

Example:
    python3 submit_backgrounds.py --input-dir input --config configs/2022.yaml
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def count_files_in_background(bkg_file: Path) -> int:
    """Count non-empty, non-comment lines in a background file."""
    count = 0
    with open(bkg_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                count += 1
    return count


def find_background_files(input_dir: Path) -> List[Path]:
    """Find all bkg_*.txt files in the input directory."""
    bkg_files = sorted(input_dir.glob('bkg_*.txt'))
    return bkg_files


def create_submit_file(
    template_file: Path,
    output_file: Path,
    bkg_file: str,
    num_jobs: int,
    config: str,
    regions_config: str,
    executor: str,
    chunk_size: int,
    workers: int,
    max_events: str = "",
) -> None:
    """Create a submit file from template with specific settings."""
    with open(template_file, 'r') as f:
        lines = f.readlines()
    
    # Build environment variable string
    env_parts = [
        f'DBL_CONFIG={config}',
        f'DBL_REGIONS_CONFIG={regions_config}',
        f'DBL_BKG_FILE={bkg_file}',
        f'DBL_EXECUTOR={executor}',
        f'DBL_CHUNK_SIZE={chunk_size}',
        f'DBL_WORKERS={workers}',
    ]
    if max_events:
        env_parts.append(f'DBL_MAX_EVENTS={max_events}')
    
    env_vars = ' \\\n'.join(env_parts)
    
    # Replace lines
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Replace environment block
        if line.strip().startswith('environment ='):
            new_lines.append(f'environment = "{env_vars}"\n')
            # Skip continuation lines until we find the closing quote
            i += 1
            while i < len(lines) and ('\\' in lines[i] or lines[i].strip().endswith('"') or not lines[i].strip()):
                i += 1
            continue
        
        # Replace queue line
        if line.strip().startswith('queue'):
            new_lines.append(f'queue {num_jobs}\n')
            i += 1
            continue
        
        new_lines.append(line)
        i += 1
    
    with open(output_file, 'w') as f:
        f.writelines(new_lines)


def submit_background(
    bkg_file: Path,
    condor_dir: Path,
    template_file: Path,
    config: str,
    regions_config: str,
    executor: str,
    chunk_size: int,
    workers: int,
    max_events: str,
    dry_run: bool = False,
) -> Tuple[str, int]:
    """Submit condor jobs for a single background file."""
    bkg_name = bkg_file.stem  # e.g., "bkg_ttbar" from "bkg_ttbar.txt"
    bkg_filename = bkg_file.name  # e.g., "bkg_ttbar.txt"
    
    # Count files in background
    num_files = count_files_in_background(bkg_file)
    
    if num_files == 0:
        print(f"⚠  Skipping {bkg_filename}: No files found")
        return bkg_name, 0
    
    print(f"\n{'='*60}")
    print(f"Background: {bkg_filename}")
    print(f"{'='*60}")
    print(f"  Files in background: {num_files}")
    print(f"  Will submit: {num_files} jobs (1 job per file)")
    
    # Create temporary submit file
    temp_submit = condor_dir / f"submit_{bkg_name}.sub"
    
    create_submit_file(
        template_file=template_file,
        output_file=temp_submit,
        bkg_file=bkg_filename,
        num_jobs=num_files,
        config=config,
        regions_config=regions_config,
        executor=executor,
        chunk_size=chunk_size,
        workers=workers,
        max_events=max_events,
    )
    
    if dry_run:
        print(f"  [DRY RUN] Would submit: condor_submit {temp_submit}")
        print(f"  [DRY RUN] Submit file created: {temp_submit}")
        return bkg_name, num_files
    
    # Submit to condor
    try:
        result = subprocess.run(
            ['condor_submit', str(temp_submit)],
            capture_output=True,
            text=True,
            check=True,
        )
        
        # Extract cluster ID from output
        cluster_id = None
        for line in result.stdout.split('\n'):
            if 'submitted to cluster' in line.lower():
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit():
                        cluster_id = part
                        break
        
        print(f"  ✓ Submitted cluster: {cluster_id}")
        print(f"  ✓ Jobs: {num_files} (one per file)")
        print(f"  ✓ Each file will run on a separate node")
        
        return bkg_name, num_files
        
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Error submitting: {e}")
        print(f"  stderr: {e.stderr}")
        return bkg_name, 0
    except FileNotFoundError:
        print(f"  ✗ Error: condor_submit not found. Are you on a condor-enabled system?")
        return bkg_name, 0


def main():
    parser = argparse.ArgumentParser(
        description='Submit condor jobs for background files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Submit all bkg_*.txt files in input/ directory
  python3 submit_backgrounds.py

  # Specify input directory
  python3 submit_backgrounds.py --input-dir input

  # Dry run (don't actually submit)
  python3 submit_backgrounds.py --dry-run

  # Custom configuration
  python3 submit_backgrounds.py --config configs/2022.yaml --workers 8
        """
    )
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('input'),
        help='Directory containing bkg_*.txt files (default: input)',
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='configs/2022.yaml',
        help='Configuration file (default: configs/2022.yaml)',
    )
    
    parser.add_argument(
        '--regions-config',
        type=str,
        default='configs/regions.yaml',
        help='Regions configuration file (default: configs/regions.yaml)',
    )
    
    parser.add_argument(
        '--executor',
        type=str,
        default='futures',
        choices=['iterative', 'futures', 'dask'],
        help='Executor type (default: futures)',
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=50000,
        help='Chunk size for futures/dask (default: 50000)',
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of workers (default: 4)',
    )
    
    parser.add_argument(
        '--max-events',
        type=str,
        default='',
        help='Maximum events to process (optional)',
    )
    
    parser.add_argument(
        '--template',
        type=Path,
        default=Path(__file__).parent / 'submit.sub',
        help='Template submit file (default: submit.sub)',
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be submitted without actually submitting',
    )
    
    args = parser.parse_args()
    
    # Get script directory
    condor_dir = Path(__file__).parent
    input_dir = args.input_dir if args.input_dir.is_absolute() else condor_dir.parent / args.input_dir
    
    # Check input directory exists
    if not input_dir.exists():
        print(f"✗ Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Check template file exists
    if not args.template.exists():
        print(f"✗ Error: Template file not found: {args.template}")
        sys.exit(1)
    
    # Find background files
    bkg_files = find_background_files(input_dir)
    
    if not bkg_files:
        print(f"✗ No bkg_*.txt files found in {input_dir}")
        sys.exit(1)
    
    print("="*60)
    print("DarkBottomLine Condor Job Submission")
    print("="*60)
    print(f"Input directory: {input_dir}")
    print(f"Found {len(bkg_files)} background file(s):")
    for bkg in bkg_files:
        num_files = count_files_in_background(bkg)
        print(f"  - {bkg.name}: {num_files} files")
    print()
    
    if args.dry_run:
        print("⚠  DRY RUN MODE - No jobs will be submitted")
        print()
    
    # Submit each background
    results = []
    for bkg_file in bkg_files:
        bkg_name, num_jobs = submit_background(
            bkg_file=bkg_file,
            condor_dir=condor_dir,
            template_file=args.template,
            config=args.config,
            regions_config=args.regions_config,
            executor=args.executor,
            chunk_size=args.chunk_size,
            workers=args.workers,
            max_events=args.max_events,
            dry_run=args.dry_run,
        )
        results.append((bkg_name, num_jobs))
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    total_jobs = 0
    for bkg_name, num_jobs in results:
        if num_jobs > 0:
            print(f"  {bkg_name}: {num_jobs} jobs")
            total_jobs += num_jobs
        else:
            print(f"  {bkg_name}: Failed or skipped")
    
    print(f"\nTotal jobs submitted: {total_jobs}")
    print("\nEach background runs on separate condor cluster.")
    print("Each file from a background runs on a separate node.")
    print("\nMonitor jobs with: condor_q")


if __name__ == '__main__':
    main()
