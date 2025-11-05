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
from typing import Dict, Any, Optional, Union

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
    events: Any,
    processor: DarkBottomLineProcessor,
    workers: int = 4
) -> Dict[str, Any]:
    """
    Run analysis using futures executor.

    Args:
        events: Events to process
        processor: Analysis processor
        workers: Number of parallel workers

    Returns:
        Analysis results
    """
    if not COFFEA_AVAILABLE:
        logging.warning("Coffea not available. Falling back to iterative execution.")
        return run_analysis_iterative(events, processor)

    logging.info(f"Running analysis with futures executor ({workers} workers)...")
    start_time = time.time()

    # Use Coffea futures executor
    executor = processor.FuturesExecutor(workers=workers)
    runner = processor.Runner(executor=executor)

    results = runner(events, processor)

    processing_time = time.time() - start_time
    logging.info(f"Futures processing completed in {processing_time:.2f} seconds")

    return results


def run_analysis_dask(
    events: Any,
    processor: DarkBottomLineProcessor,
    workers: int = 4
) -> Dict[str, Any]:
    """
    Run analysis using Dask executor.

    Args:
        events: Events to process
        processor: Analysis processor
        workers: Number of parallel workers

    Returns:
        Analysis results
    """
    if not COFFEA_AVAILABLE:
        logging.warning("Coffea not available. Falling back to iterative execution.")
        return run_analysis_iterative(events, processor)

    try:
        from dask.distributed import Client
        from coffea.processor import DaskExecutor

        logging.info(f"Running analysis with Dask executor ({workers} workers)...")
        start_time = time.time()

        # Start Dask client
        client = Client(n_workers=workers)

        try:
            # Use Coffea Dask executor
            executor = DaskExecutor(client=client)
            runner = processor.Runner(executor=executor)

            results = runner(events, processor)

            processing_time = time.time() - start_time
            logging.info(f"Dask processing completed in {processing_time:.2f} seconds")

            return results

        finally:
            client.close()

    except ImportError:
        logging.warning("Dask not available. Falling back to iterative execution.")
        return run_analysis_iterative(events, processor)


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
    parser.add_argument("--max-events", type=int, default=None,
                       help="Maximum number of events to process")
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

    # Load events
    logging.info(f"Loading events from {args.input}")
    events = load_events(args.input, args.max_events)

    # Initialize processor
    logging.info("Initializing processor...")
    processor = DarkBottomLineProcessor(config)

    # Run analysis
    logging.info(f"Starting analysis with {args.executor} executor...")
    start_time = time.time()

    if args.executor == "iterative":
        results = run_analysis_iterative(events, processor)
    elif args.executor == "futures":
        results = run_analysis_futures(events, processor, args.workers)
    elif args.executor == "dask":
        results = run_analysis_dask(events, processor, args.workers)
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
