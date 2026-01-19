#!/usr/bin/env python3
"""
Main execution script for DarkBottomLine framework.
"""

import argparse
import logging
import time
import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

try:
    from coffea import processor
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    COFFEA_AVAILABLE = True
except ImportError:
    COFFEA_AVAILABLE = False
    logging.warning("Coffea not available. Using fallback implementation.")

from darkbottomline.processor import DarkBottomLineProcessor, DarkBottomLineCoffeaProcessor


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Add command line options to config
    config['save_skims'] = True  # Default to saving skims

    return config


def load_events(input_path: str, max_events: Optional[int] = None) -> Any:
    """
    Load events from input file.

    Args:
        input_path: Path to input file
        max_events: Maximum number of events to process

    Returns:
        Events array
    """
    if COFFEA_AVAILABLE:
        # Use Coffea NanoEvents
        events = NanoEventsFactory.from_root(
            input_path,
            schemaclass=NanoAODSchema,
            maxchunks=max_events
        ).events()
        return events
    else:
        # Fallback to uproot
        try:
            import uproot
            with uproot.open(input_path) as f:
                events = f["Events"].arrays()
            if max_events:
                events = events[:max_events]
            return events
        except ImportError:
            raise ImportError("Neither Coffea nor uproot available. Cannot load events.")


def run_analysis_iterative(
    events: Any,
    processor: DarkBottomLineProcessor
) -> Dict[str, Any]:
    """
    Run analysis using iterative execution.

    Args:
        events: Events to process
        processor: Analysis processor

    Returns:
        Analysis results
    """
    logging.info("Running analysis iteratively...")
    start_time = time.time()

    results = processor.process(events)

    processing_time = time.time() - start_time
    logging.info(f"Iterative processing completed in {processing_time:.2f} seconds")

    return results


def run_analysis_futures(
    fileset: Dict[str, List[str]],
    processor_instance: DarkBottomLineCoffeaProcessor,
    workers: int = 4,
    chunksize: int = 50000,
    maxchunks: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run analysis using futures executor with run_uproot_job.

    Args:
        fileset: Dictionary mapping dataset names to file paths
        processor_instance: Coffea-compatible processor instance
        workers: Number of parallel workers
        chunksize: Number of events per chunk
        maxchunks: Optional limit on number of chunks to process

    Returns:
        Analysis results
    """
    if not COFFEA_AVAILABLE:
        logging.warning("Coffea not available. Falling back to iterative execution.")
        return None

    from coffea.processor import run_uproot_job, FuturesExecutor

    logging.info(f"Running analysis with futures executor ({workers} workers, chunksize={chunksize})...")
    if maxchunks:
        logging.info(f"Limiting to {maxchunks} chunks")
    start_time = time.time()

    executor = FuturesExecutor(workers=workers)

    result = run_uproot_job(
        fileset,
        "Events",
        processor_instance,
        executor=executor,
        executor_args={"workers": workers, "chunksize": chunksize},
        chunksize=chunksize,
        maxchunks=maxchunks,
    )

    processing_time = time.time() - start_time
    logging.info(f"Futures processing completed in {processing_time:.2f} seconds")

    return result


def run_analysis_dask(
    fileset: Dict[str, List[str]],
    processor_instance: DarkBottomLineCoffeaProcessor,
    workers: int = 4,
    chunksize: int = 200000,
    maxchunks: Optional[int] = None
) -> Dict[str, Any]:
    """
    Run analysis using Dask executor with run_uproot_job.

    Args:
        fileset: Dictionary mapping dataset names to file paths
        processor_instance: Coffea-compatible processor instance
        workers: Number of parallel workers
        chunksize: Number of events per chunk
        maxchunks: Optional limit on number of chunks to process

    Returns:
        Analysis results
    """
    if not COFFEA_AVAILABLE:
        logging.warning("Coffea not available. Falling back to iterative execution.")
        return None

    try:
        from dask.distributed import Client
        from coffea.processor import run_uproot_job, DaskExecutor

        logging.info(f"Running analysis with Dask executor ({workers} workers, chunksize={chunksize})...")
        if maxchunks:
            logging.info(f"Limiting to {maxchunks} chunks")
        start_time = time.time()

        # Start Dask client
        client = Client(n_workers=workers)

        try:
            executor = DaskExecutor(client=client)

            result = run_uproot_job(
                fileset,
                "Events",
                processor_instance,
                executor=executor,
                executor_args={"client": client, "flatten": True, "retries": 3},
                chunksize=chunksize,
                maxchunks=maxchunks,
            )

            processing_time = time.time() - start_time
            logging.info(f"Dask processing completed in {processing_time:.2f} seconds")

            return result

        finally:
            client.close()

    except ImportError:
        logging.warning("Dask not available. Falling back to iterative execution.")
        return None


def save_results(results: Dict[str, Any], output_path: str):
    """
    Save analysis results to file.

    Args:
        results: Analysis results
        output_path: Output file path
    """
    logging.info(f"Saving results to {output_path}")

    if output_path.endswith('.parquet'):
        save_parquet_results(results, output_path)
    elif output_path.endswith('.root'):
        save_root_results(results, output_path)
    else:
        save_pickle_results(results, output_path)


def save_parquet_results(results: Dict[str, Any], output_path: str):
    """Save results as Parquet file."""
    try:
        import pandas as pd
        import awkward as ak

        # Convert histograms to DataFrame format
        data = {}
        for name, hist in results.get("histograms", {}).items():
            if hasattr(hist, 'values'):
                data[name] = hist.values().flatten()
            else:
                data[name] = hist.get("values", [])

        df = pd.DataFrame(data)
        df.to_parquet(output_path)
        logging.info(f"Saved results to {output_path}")

    except ImportError:
        logging.warning("pandas not available. Falling back to pickle.")
        save_pickle_results(results, output_path)


def save_root_results(results: Dict[str, Any], output_path: str):
    """Save results as ROOT file."""
    try:
        import uproot

        with uproot.recreate(output_path) as f:
            # Save histograms
            for name, hist in results.get("histograms", {}).items():
                if hasattr(hist, 'values'):
                    f[name] = hist

            # Save metadata
            f["metadata"] = results.get("metadata", {})

        logging.info(f"Saved results to {output_path}")

    except ImportError:
        logging.warning("uproot not available. Falling back to pickle.")
        save_pickle_results(results, output_path)


def save_pickle_results(results: Dict[str, Any], output_path: str):
    """Save results as pickle file."""
    import pickle

    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    logging.info(f"Saved results to {output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="DarkBottomLine Analysis Framework")

    # Required arguments
    parser.add_argument("--config", required=True, help="Path to configuration YAML file")
    parser.add_argument("--input", required=True, help="Path to input NanoAOD file")
    parser.add_argument("--output", required=True, help="Path to output file")

    # Optional arguments
    parser.add_argument("--executor", choices=["iterative", "futures", "dask"],
                       default="iterative", help="Execution backend")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--chunk-size", type=int, default=50000,
                       help="Number of events per chunk for futures/dask executors (default: 50000)")
    parser.add_argument("--max-events", type=int, default=None,
                       help="Maximum number of events to process (converted to maxchunks for run_uproot_job)")
    parser.add_argument("--save-skims", action="store_true",
                       help="Save skimmed events")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Load configuration
    logging.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override config with command line arguments
    config['save_skims'] = args.save_skims

    # Initialize processor
    logging.info("Initializing processor...")
    base_processor = DarkBottomLineProcessor(config)

    # Run analysis
    logging.info(f"Starting analysis with {args.executor} executor...")
    start_time = time.time()

    if args.executor == "iterative":
        # Load events for iterative mode
        logging.info(f"Loading events from {args.input}")
        events = load_events(args.input, args.max_events)
        results = run_analysis_iterative(events, base_processor)
    elif args.executor == "futures":
        # Use run_uproot_job with FuturesExecutor
        input_files = []
        if args.input.endswith('.txt'):
            with open(args.input, 'r') as f:
                input_files = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        else:
            input_files = [args.input]
        
        fileset = {"dataset": input_files}
        chunksize = args.chunk_size
        maxchunks = None
        if args.max_events:
            # Calculate maxchunks based on max_events and chunksize
            maxchunks = (args.max_events + chunksize - 1) // chunksize
        
        coffea_processor = DarkBottomLineCoffeaProcessor(config)
        results = run_analysis_futures(fileset, coffea_processor, args.workers, chunksize, maxchunks)
    elif args.executor == "dask":
        # Use run_uproot_job with DaskExecutor
        input_files = []
        if args.input.endswith('.txt'):
            with open(args.input, 'r') as f:
                input_files = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        else:
            input_files = [args.input]
        
        fileset = {"dataset": input_files}
        chunksize = args.chunk_size if args.chunk_size != 50000 else 200000  # Default to 200k for dask
        maxchunks = None
        if args.max_events:
            maxchunks = (args.max_events + chunksize - 1) // chunksize
        
        coffea_processor = DarkBottomLineCoffeaProcessor(config)
        results = run_analysis_dask(fileset, coffea_processor, args.workers, chunksize, maxchunks)
    else:
        raise ValueError(f"Unknown executor: {args.executor}")

    total_time = time.time() - start_time
    logging.info(f"Analysis completed in {total_time:.2f} seconds")

    # Save results
    save_results(results, args.output)

    # Print summary
    if "cutflow" in results:
        print("\n" + processor.get_cutflow_summary())

    if "metadata" in results:
        print("\n" + processor.get_processing_summary())

    logging.info("Analysis completed successfully!")


if __name__ == "__main__":
    main()
