"""
Main processor class for DarkBottomLine framework.
"""

import awkward as ak
import numpy as np
from typing import Dict, Any, Optional, Union
import logging
import time

try:
    from coffea import processor
    from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
    COFFEA_AVAILABLE = True
except ImportError:
    COFFEA_AVAILABLE = False
    logging.warning("Coffea not available. Using fallback implementation.")

from .objects import build_objects
from .selections import apply_selection
from .corrections import CorrectionManager
from .weights import WeightCalculator
from .histograms import HistogramManager


class DarkBottomLineProcessor:
    """
    Main processor class for DarkBottomLine analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the processor with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.correction_manager = CorrectionManager(config)
        self.histogram_manager = HistogramManager()
        self.histograms = self.histogram_manager.define_histograms()

        # Initialize accumulators
        self.accumulator = {
            "histograms": self.histograms,
            "cutflow": {},
            "metadata": {},
        }

        logging.info(f"Initialized DarkBottomLine processor for year {config.get('year', 'unknown')}")

    def process(self, events: ak.Array) -> Dict[str, Any]:
        """
        Process events through the analysis chain.

        Args:
            events: Awkward Array of events

        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        print(f"=== PROCESSING EVENTS ===")
        print(f"Total events loaded: {len(events)}")

        # Build physics objects
        print("Step 1: Building physics objects...")
        logging.info("Building physics objects...")
        objects = build_objects(events, self.config)

        # Print object counts
        n_muons = ak.sum(ak.num(objects["muons"], axis=1))
        n_electrons = ak.sum(ak.num(objects["electrons"], axis=1))
        n_taus = ak.sum(ak.num(objects["taus"], axis=1))
        n_jets = ak.sum(ak.num(objects["jets"], axis=1))
        n_bjets = ak.sum(ak.num(objects["bjets"], axis=1))
        print(f"  Selected objects: {n_muons} muons, {n_electrons} electrons, {n_taus} taus, {n_jets} jets, {n_bjets} b-jets")

        # Apply event selection
        print("Step 2: Applying event selection...")
        logging.info("Applying event selection...")
        selected_events, selected_objects, cutflow = apply_selection(
            events, objects, self.config
        )
        print(f"  Events passing selection: {len(selected_events)} / {len(events)} ({len(selected_events)/len(events)*100:.1f}%)")

        # Calculate corrections and weights
        print("Step 3: Calculating corrections and weights...")
        print(f"  Selected events for weight calculation: {len(selected_events)}")
        logging.info("Calculating corrections and weights...")

        try:
            corrections = self.correction_manager.get_all_corrections(
                selected_events, selected_objects
            )

            # Initialize weight calculator
            weight_calculator = WeightCalculator(len(selected_events))

            # Add generator weights
            weight_calculator.add_generator_weight(selected_events)

            # Add corrections
            weight_calculator.add_corrections(corrections)

            # Get final weights
            event_weights = weight_calculator.get_weight("central")
            print(f"  Weights calculated successfully: {len(event_weights)} weights")
        except Exception as e:
            print(f"  Error in weight calculation: {e}")
            print(f"  Selected events length: {len(selected_events)}")
            raise

        # Fill histograms
        print("Step 4: Filling histograms...")
        print(f"  Selected events: {len(selected_events)}")
        print(f"  Event weights shape: {event_weights.shape if hasattr(event_weights, 'shape') else len(event_weights)}")
        logging.info("Filling histograms...")
        try:
            filled_histograms = self.histogram_manager.fill_histograms(
                selected_events, selected_objects, event_weights
            )
            print(f"  Histograms filled: {list(filled_histograms.keys())}")
        except Exception as e:
            print(f"  Error in histogram filling: {e}")
            print(f"  Selected events length: {len(selected_events)}")
            print(f"  Objects keys: {list(selected_objects.keys())}")
            for key, obj in selected_objects.items():
                if isinstance(obj, ak.Array):
                    print(f"    {key}: {len(obj)} events, {ak.num(obj, axis=1).sum()} total objects")
            raise

        # Calculate processing statistics
        processing_time = time.time() - start_time

        # Update accumulator
        self.accumulator["histograms"] = filled_histograms
        self.accumulator["cutflow"] = cutflow
        self.accumulator["metadata"] = {
            "n_events_processed": len(events),
            "n_events_selected": len(selected_events),
            "processing_time": processing_time,
            "weight_statistics": weight_calculator.get_weight_statistics(),
        }

        # Optionally save skimmed events
        if self.config.get("save_skims", False):
            print("Step 5: Creating skimmed events...")
            self.accumulator["skimmed_events"] = self._create_skimmed_events(
                selected_events, selected_objects, event_weights
            )

        print("=== ANALYSIS COMPLETE ===")
        print(f"Processed {len(events)} events, selected {len(selected_events)} events")
        print(f"Processing time: {processing_time:.2f} seconds")
        logging.info(f"Processed {len(events)} events, selected {len(selected_events)} events")
        logging.info(f"Processing time: {processing_time:.2f} seconds")

        return self.accumulator

    def _create_skimmed_events(
        self,
        events: ak.Array,
        objects: Dict[str, Any],
        weights: Union[ak.Array, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Create skimmed events for downstream analysis.

        Args:
            events: Selected events
            objects: Selected objects
            weights: Event weights

        Returns:
            Dictionary containing skimmed event data
        """
        skimmed = {
            "event": events.event,
            "run": events.run,
            "luminosityBlock": events.luminosityBlock,
            "MET": {
                "pt": events["PFMET_pt"],
                "phi": events["PFMET_phi"],
                "significance": events["PFMET_significance"],
            },
            "weights": weights,
        }

        # Add selected objects
        for obj_name, obj_data in objects.items():
            if isinstance(obj_data, ak.Array):
                skimmed[obj_name] = obj_data

        return skimmed

    def get_histogram_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all histograms.

        Returns:
            Dictionary with statistics for each histogram
        """
        return self.histogram_manager.get_histogram_statistics(self.histograms)

    def save_results(self, output_file: str):
        """
        Save analysis results to file.

        Args:
            output_file: Output file path
        """
        if output_file.endswith('.parquet'):
            self._save_parquet(output_file)
        elif output_file.endswith('.root'):
            self._save_root(output_file)
        else:
            self._save_pickle(output_file)

    def _save_parquet(self, output_file: str):
        """Save results as Parquet file."""
        import pandas as pd

        # Convert histograms to DataFrame format
        data = {}
        for name, hist in self.accumulator["histograms"].items():
            if hasattr(hist, 'values'):
                data[name] = hist.values().flatten()
            else:
                data[name] = hist.get("values", [])

        df = pd.DataFrame(data)
        df.to_parquet(output_file)
        logging.info(f"Saved results to {output_file}")

    def _save_root(self, output_file: str):
        """Save results as ROOT file."""
        try:
            import uproot
            with uproot.recreate(output_file) as f:
                # Save histograms
                for name, hist in self.accumulator["histograms"].items():
                    if hasattr(hist, 'values'):
                        f[name] = hist

                # Save metadata
                f["metadata"] = self.accumulator["metadata"]

            logging.info(f"Saved results to {output_file}")
        except ImportError:
            logging.warning("uproot not available. Falling back to pickle.")
            self._save_pickle(output_file)

    def _save_pickle(self, output_file: str):
        """Save results as pickle file."""
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(self.accumulator, f)
        logging.info(f"Saved results to {output_file}")

    def get_cutflow_summary(self) -> str:
        """
        Get a formatted cutflow summary.

        Returns:
            Formatted cutflow string
        """
        cutflow = self.accumulator.get("cutflow", {})
        if not cutflow:
            return "No cutflow data available"

        summary = "Cutflow Summary:\n"
        summary += "=" * 50 + "\n"

        for cut_name, n_events in cutflow.items():
            summary += f"{cut_name:20s}: {n_events:8d}\n"

        return summary

    def get_processing_summary(self) -> str:
        """
        Get a formatted processing summary.

        Returns:
            Formatted processing summary string
        """
        metadata = self.accumulator.get("metadata", {})
        if not metadata:
            return "No processing data available"

        summary = "Processing Summary:\n"
        summary += "=" * 50 + "\n"
        summary += f"Events processed: {metadata.get('n_events_processed', 0)}\n"
        summary += f"Events selected:  {metadata.get('n_events_selected', 0)}\n"
        summary += f"Processing time:   {metadata.get('processing_time', 0):.2f} seconds\n"

        weight_stats = metadata.get('weight_statistics', {})
        if weight_stats:
            summary += f"Weight statistics:\n"
            summary += f"  Mean: {weight_stats.get('mean', 0):.4f}\n"
            summary += f"  Std:  {weight_stats.get('std', 0):.4f}\n"
            summary += f"  Sum:  {weight_stats.get('sum', 0):.2f}\n"

        return summary


# Coffea processor wrapper for compatibility
if COFFEA_AVAILABLE:
    class DarkBottomLineCoffeaProcessor(processor.ProcessorABC):
        """
        Coffea-compatible processor wrapper.
        """

        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.processor = DarkBottomLineProcessor(config)

        def process(self, events: ak.Array) -> Dict[str, Any]:
            """Process events using the main processor."""
            return self.processor.process(events)

        def postprocess(self, accumulator: Dict[str, Any]) -> Dict[str, Any]:
            """Post-process results."""
            return accumulator
