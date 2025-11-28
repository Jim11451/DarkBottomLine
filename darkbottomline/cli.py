"""
Command-line interface for DarkBottomLine framework.
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from .processor import DarkBottomLineProcessor
from .analyzer import DarkBottomLineAnalyzer
from .dnn_trainer import DNNTrainer
from .dnn_inference import DNNInference
from .plotting import PlotManager
from .regions import RegionManager


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_analysis(args):
    """Run basic analysis."""
    logging.info("Running basic analysis...")

    # Load configuration
    config = load_config(args.config)

    # Initialize processor
    processor = DarkBottomLineProcessor(config)

    # Load events from ROOT file
    try:
        import uproot
        import awkward as ak

        logging.info(f"Loading events from {args.input}")
        with uproot.open(args.input) as f:
            # Load all branches
            events = f["Events"].arrays()

        # Limit events if specified
        if args.max_events and len(events) > args.max_events:
            events = events[:args.max_events]
            logging.info(f"Limited to {args.max_events} events")

        logging.info(f"Loaded {len(events)} events")

        # Process events (optionally save event-level selection)
        results = processor.process(events, event_selection_output=args.event_selection_output)

        # Save results
        import pickle
        import os

        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output), exist_ok=True)

        with open(args.output, 'wb') as f:
            pickle.dump(results, f)

        logging.info(f"Results saved to {args.output}")

    except Exception as e:
        logging.error(f"Error processing events: {e}")
        raise

    logging.info("Basic analysis completed!")


def run_analyzer(args):
    """Run multi-region analysis."""
    logging.info("Running multi-region analysis...")

    # Load configuration
    config = load_config(args.config)

    # Initialize analyzer
    analyzer = DarkBottomLineAnalyzer(config, args.regions_config)

    try:
        import uproot
        import awkward as ak

        logging.info(f"Loading events from {args.input}")
        with uproot.open(args.input) as f:
            events = f["Events"].arrays()

        # Limit events if specified
        if args.max_events and len(events) > args.max_events:
            events = events[:args.max_events]
            logging.info(f"Limited to {args.max_events} events")

        logging.info(f"Loaded {len(events)} events")

        # Process events through regions (optionally save event-level selection)
        results = analyzer.process(events, event_selection_output=args.event_selection_output)

        # Save results
        import os
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        analyzer.accumulator = results
        analyzer.save_results(args.output)

    except Exception as e:
        logging.error(f"Error in multi-region analysis: {e}")
        raise

    logging.info("Multi-region analysis completed!")


def train_dnn(args):
    """Train DNN model."""
    logging.info("Training DNN model...")

    # Initialize trainer
    trainer = DNNTrainer(args.config)

    # Load data
    signal_files = args.signal_files
    background_files = args.background_files

    features, labels, masses = trainer.load_data(signal_files, background_files)

    # Preprocess data
    train_features, train_labels, train_masses, val_features, val_labels, val_masses = trainer.preprocess(features, labels, masses)

    # Train model
    results = trainer.train(train_features, train_labels, train_masses, val_features, val_labels, val_masses)

    # Save model
    trainer.save_model(args.output)

    # Plot training history
    if args.plot_history:
        trainer.plot_training_history(f"{args.output}_history.png")

    logging.info("DNN training completed!")


def make_plots(args):
    """Create data/MC plots."""
    logging.info("Creating plots...")

    # Load results
    import pickle
    with open(args.input, 'rb') as f:
        results = pickle.load(f)

    # Load plotting config if provided
    plot_config = None
    if args.plot_config:
        plot_config = load_config(args.plot_config)
        logging.info(f"Loaded plotting configuration from {args.plot_config}")
    else:
        # Try to load default plotting config
        default_plot_config_path = Path(__file__).parent.parent / "configs" / "plotting.yaml"
        if default_plot_config_path.exists():
            plot_config = load_config(str(default_plot_config_path))
            logging.info(f"Loaded default plotting configuration from {default_plot_config_path}")

    # Initialize plot manager with config
    plot_manager = PlotManager(config=plot_config)

    # Generate version string if not provided (format: YYYYMMDD_HHMM)
    if not args.version:
        from datetime import datetime
        version = datetime.now().strftime("%Y%m%d_%H%M")
    else:
        version = args.version

    # Create output directory
    import os
    os.makedirs(args.save_dir, exist_ok=True)

    # Create plots with all formats automatically (PNG, PDF, ROOT, TXT)
    plot_files = plot_manager.create_all_plots(
        results, args.save_dir, args.show_data, args.regions, version, formats=None
    )

    logging.info(f"Plots saved to {args.save_dir}")
    logging.info("Plot creation completed!")


def make_stacked_plots(args):
    """Create stacked Data/MC plots with ratio and uncertainty band."""
    logging.info("Creating stacked plots...")

    # Load plotting config if provided
    plot_config = None
    if args.plot_config:
        plot_config = load_config(args.plot_config)
        logging.info(f"Loaded plotting configuration from {args.plot_config}")
    else:
        # Try to load default plotting config
        default_plot_config_path = Path(__file__).parent.parent / "configs" / "plotting.yaml"
        if default_plot_config_path.exists():
            plot_config = load_config(str(default_plot_config_path))
            logging.info(f"Loaded default plotting configuration from {default_plot_config_path}")

    plot_manager = PlotManager(config=plot_config)

    # Generate version string if not provided (format: YYYYMMDD_HHMM)
    if not args.version:
        from datetime import datetime
        version = datetime.now().strftime("%Y%m%d_%H%M")
    else:
        version = args.version

    # Parse inputs
    data_file = args.data
    bkg_files = args.backgrounds or []
    signal_file = args.signal
    output = args.output
    variable = args.variable
    xlabel = args.xlabel
    title_tag = args.title

    # Run with multi-format saving
    out = plot_manager.create_stacked_plot_from_files(
        data_file=data_file,
        background_files=bkg_files,
        signal_file=signal_file,
        output_path=output,
        variable=variable,
        xlabel=xlabel,
        title_tag=title_tag,
        version=version,
        formats=None  # All formats generated automatically
    )

    logging.info(f"Stacked plot saved to {out}")


def make_datacard(args):
    """Generate Combine datacard."""
    logging.info("Generating datacard...")

    # Load results
    # results = load_results(args.input)

    # Generate datacard (placeholder)
    # datacard_writer = CombineDatacardWriter()
    # datacard_writer.write_datacard(results, args.output)

    logging.info("Datacard generation completed!")


def run_combine(args):
    """Run Combine fits."""
    logging.info("Running Combine fits...")

    # Run Combine command (placeholder)
    # combine_runner = CombineRunner()
    # results = combine_runner.run_fit(args.mode, args.datacard, args.options)

    logging.info("Combine execution completed!")


def make_impact(args):
    """Create impact plots."""
    logging.info("Creating impact plots...")

    # Load fit results
    # results = load_fit_results(args.input)

    # Create impact plots (placeholder)
    # diagnostic_plotter = DiagnosticPlotter()
    # diagnostic_plotter.plot_impacts(results, args.output)

    logging.info("Impact plot creation completed!")


def make_pulls(args):
    """Create pull plots."""
    logging.info("Creating pull plots...")

    # Load fit results
    # results = load_fit_results(args.input)

    # Create pull plots (placeholder)
    # diagnostic_plotter = DiagnosticPlotter()
    # diagnostic_plotter.plot_pulls(results, args.output)

    logging.info("Pull plot creation completed!")


def make_gof(args):
    """Create goodness-of-fit plots."""
    logging.info("Creating GOF plots...")

    # Load GOF results
    # results = load_gof_results(args.input)

    # Create GOF plots (placeholder)
    # diagnostic_plotter = DiagnosticPlotter()
    # diagnostic_plotter.plot_gof(results, args.output)

    logging.info("GOF plot creation completed!")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="DarkBottomLine Framework - Advanced Analysis Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run basic analysis
  darkbottomline run --config configs/2023.yaml --input data.root --output results.coffea

  # Run multi-region analysis
  darkbottomline analyze --config configs/2023.yaml --regions-config configs/regions.yaml --input data.root --output results.coffea

  # Train DNN
  darkbottomline train-dnn --config configs/dnn.yaml --signal signals.root --background bkg.root --output model.pt

  # Create plots
  darkbottomline make-plots --year 2023 --region SR --show-data False --save-dir outputs/plots/

  # Generate datacard
  darkbottomline make-datacard --region SR --output outputs/combine/ --year 2023

  # Run Combine fits
  darkbottomline run-combine --mode FitDiagnostics --datacard outputs/combine/datacard.txt

  # Create diagnostic plots
  darkbottomline make-impact --input outputs/combine/fitDiagnostics.root --output outputs/plots/
        """
    )

    # Global arguments
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run basic analysis")
    run_parser.add_argument("--config", required=True, help="Configuration file")
    run_parser.add_argument("--input", required=True, help="Input file")
    run_parser.add_argument("--output", required=True, help="Output file")
    run_parser.add_argument("--event-selection-output", help="Path to save events that pass event-level selection (optional)")
    run_parser.add_argument("--executor", choices=["iterative", "futures", "dask"],
                           default="iterative", help="Execution backend")
    run_parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    run_parser.add_argument("--max-events", type=int, help="Maximum events to process")
    run_parser.set_defaults(func=run_analysis)

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Run multi-region analysis")
    analyze_parser.add_argument("--config", required=True, help="Base configuration file")
    analyze_parser.add_argument("--regions-config", required=True, help="Regions configuration file")
    analyze_parser.add_argument("--input", required=True, help="Input file")
    analyze_parser.add_argument("--output", required=True, help="Output file")
    analyze_parser.add_argument("--event-selection-output", help="Path to save events that pass event-level selection (optional)")
    analyze_parser.add_argument("--executor", choices=["iterative", "futures", "dask"],
                               default="iterative", help="Execution backend")
    analyze_parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    analyze_parser.add_argument("--max-events", type=int, help="Maximum events to process")
    analyze_parser.set_defaults(func=run_analyzer)

    # Train DNN command
    train_dnn_parser = subparsers.add_parser("train-dnn", help="Train DNN model")
    train_dnn_parser.add_argument("--config", required=True, help="DNN configuration file")
    train_dnn_parser.add_argument("--signal", nargs="+", required=True, help="Signal files")
    train_dnn_parser.add_argument("--background", nargs="+", required=True, help="Background files")
    train_dnn_parser.add_argument("--output", required=True, help="Output model file")
    train_dnn_parser.add_argument("--plot-history", action="store_true", help="Plot training history")
    train_dnn_parser.set_defaults(func=train_dnn)

    # Make plots command
    plots_parser = subparsers.add_parser("make-plots", help="Create data/MC plots")
    plots_parser.add_argument("--input", required=True, help="Input results file")
    plots_parser.add_argument("--save-dir", required=True, help="Output directory")
    plots_parser.add_argument("--year", help="Data-taking year")
    plots_parser.add_argument("--region", help="Specific region to plot")
    plots_parser.add_argument("--show-data", action="store_true", help="Show data points")
    plots_parser.add_argument("--regions", nargs="+", help="List of regions to plot")
    plots_parser.add_argument("--version", help="Version string (default: auto-generate timestamp)")
    plots_parser.add_argument("--plot-config", help="Path to plotting configuration YAML file (default: configs/plotting.yaml)")
    # All formats (PNG, PDF, ROOT, TXT) are generated automatically in batch mode
    plots_parser.set_defaults(func=make_plots)

    # Make stacked plots command
    stacked_parser = subparsers.add_parser("make-stacked-plots", help="Create stacked Data/MC plots with ratio")
    stacked_parser.add_argument("--data", help="Data results pickle path")
    stacked_parser.add_argument("--backgrounds", nargs="+", help="Background results pickle paths")
    stacked_parser.add_argument("--signal", help="Signal results pickle path")
    stacked_parser.add_argument("--output", required=True, help="Output plot file (e.g. outputs/plots/stacked_met.pdf)")
    stacked_parser.add_argument("--variable", default="met", help="Variable key to plot (default: met)")
    stacked_parser.add_argument("--xlabel", default="MET [GeV]", help="X-axis label")
    stacked_parser.add_argument("--title", default="CMS Preliminary  (13.6 TeV, 2023)", help="Title tag with CMS text")
    stacked_parser.add_argument("--version", help="Version string (default: auto-generate timestamp)")
    stacked_parser.add_argument("--plot-config", help="Path to plotting configuration YAML file (default: configs/plotting.yaml)")
    # All formats (PNG, PDF, ROOT, TXT) are generated automatically in batch mode
    stacked_parser.set_defaults(func=make_stacked_plots)

    # Make datacard command
    datacard_parser = subparsers.add_parser("make-datacard", help="Generate Combine datacard")
    datacard_parser.add_argument("--input", required=True, help="Input results file")
    datacard_parser.add_argument("--output", required=True, help="Output directory")
    datacard_parser.add_argument("--region", help="Specific region for datacard")
    datacard_parser.add_argument("--year", help="Data-taking year")
    datacard_parser.set_defaults(func=make_datacard)

    # Run Combine command
    combine_parser = subparsers.add_parser("run-combine", help="Run Combine fits")
    combine_parser.add_argument("--mode", required=True,
                               choices=["AsymptoticLimits", "FitDiagnostics", "GoodnessOfFit"],
                               help="Combine mode")
    combine_parser.add_argument("--datacard", required=True, help="Datacard file")
    combine_parser.add_argument("--output", help="Output directory")
    combine_parser.add_argument("--fit-region", help="Fit region")
    combine_parser.add_argument("--include-signal", action="store_true", help="Include signal in fit")
    combine_parser.add_argument("--toys", type=int, help="Number of toys for GOF")
    combine_parser.set_defaults(func=run_combine)

    # Make impact command
    impact_parser = subparsers.add_parser("make-impact", help="Create impact plots")
    impact_parser.add_argument("--input", required=True, help="Input fit results file")
    impact_parser.add_argument("--output", required=True, help="Output directory")
    impact_parser.set_defaults(func=make_impact)

    # Make pulls command
    pulls_parser = subparsers.add_parser("make-pulls", help="Create pull plots")
    pulls_parser.add_argument("--input", required=True, help="Input fit results file")
    pulls_parser.add_argument("--output", required=True, help="Output directory")
    pulls_parser.set_defaults(func=make_pulls)

    # Make GOF command
    gof_parser = subparsers.add_parser("make-gof", help="Create goodness-of-fit plots")
    gof_parser.add_argument("--input", required=True, help="Input GOF results file")
    gof_parser.add_argument("--output", required=True, help="Output directory")
    gof_parser.set_defaults(func=make_gof)

    # Parse arguments
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Check if command was provided
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Execute command
    try:
        args.func(args)
    except Exception as e:
        logging.error(f"Command failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
