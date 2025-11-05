"""
Multi-region analyzer for DarkBottomLine framework.
"""

import awkward as ak
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

from .processor import DarkBottomLineProcessor
from .objects import build_objects
from .regions import RegionManager
from .histograms import HistogramManager


class DarkBottomLineAnalyzer:
    """
    Multi-region analyzer extending the base processor.
    """

    def __init__(self, config: Dict[str, Any], regions_config_path: str):
        """
        Initialize analyzer with configuration and regions.

        Args:
            config: Base configuration dictionary
            regions_config_path: Path to regions configuration file
        """
        # Initialize base processor
        self.base_processor = DarkBottomLineProcessor(config)

        # Initialize region manager
        self.region_manager = RegionManager(regions_config_path)

        # Initialize histogram manager for regions
        self.histogram_manager = HistogramManager()

        # Create region-specific histograms
        self.region_histograms = self._create_region_histograms()

        # Initialize accumulator
        self.accumulator = {
            "regions": {},
            "region_histograms": self.region_histograms,
            "region_cutflow": {},
            "region_validation": {},
            "metadata": {}
        }

        logging.info(f"Initialized analyzer with {len(self.region_manager.regions)} regions")

    def _create_region_histograms(self) -> Dict[str, Dict[str, Any]]:
        """
        Create histograms for each region.

        Returns:
            Dictionary of region histograms
        """
        region_histograms = {}

        for region_name in self.region_manager.regions.keys():
            # Create standard histograms for each region
            histograms = self.histogram_manager.define_histograms()

            # Add region-specific histograms
            histograms[f"{region_name}_dnn_score"] = self._create_dnn_histogram()
            histograms[f"{region_name}_region_variables"] = self._create_region_variables_histogram()

            region_histograms[region_name] = histograms

        return region_histograms

    def _create_dnn_histogram(self) -> Any:
        """
        Create DNN score histogram.

        Returns:
            DNN score histogram
        """
        try:
            import hist
            return hist.Hist(
                hist.axis.Regular(50, 0, 1, name="dnn_score", label="DNN Score"),
                storage=hist.storage.Weight()
            )
        except ImportError:
            return {
                "bins": np.linspace(0, 1, 51),
                "label": "DNN Score",
                "values": [],
                "weights": []
            }

    def _create_region_variables_histogram(self) -> Any:
        """
        Create region-specific variables histogram.

        Returns:
            Region variables histogram
        """
        try:
            import hist
            return hist.Hist(
                hist.axis.Regular(50, 0, 500, name="met", label="MET [GeV]"),
                hist.axis.Regular(10, 0, 10, name="nbjets", label="Number of B-jets"),
                storage=hist.storage.Weight()
            )
        except ImportError:
            return {
                "bins": [np.linspace(0, 500, 51), np.arange(0, 11)],
                "label": "Region Variables",
                "values": [],
                "weights": []
            }

    def process(self, events: ak.Array) -> Dict[str, Any]:
        """
        Process events through all regions.

        Args:
            events: Awkward Array of events

        Returns:
            Analysis results with per-region histograms
        """
        start_time = time.time()

        # Build physics objects directly for regioning
        logging.info("Building physics objects for regions...")
        objects = build_objects(events, self.base_processor.config)

        # Apply region cuts
        logging.info("Applying region cuts...")
        region_masks = self.region_manager.apply_regions(events, objects)

        # Process each region
        region_results = {}
        for region_name, region_mask in region_masks.items():
            # Skip processing if no events pass cuts
            n_events = ak.sum(region_mask)
            if n_events == 0:
                logging.info(f"Skipping region {region_name}: 0 events")
                region_results[region_name] = {
                    "n_events": 0,
                    "variables": {},
                    "dnn_scores": None
                }
                continue

            logging.info(f"Processing region {region_name}...")
            region_results[region_name] = self._process_region(
                events, objects, region_mask, region_name
            )

        # Validate regions
        logging.info("Validating regions...")
        validation_results = self.region_manager.validate_regions(events, objects)

        # Calculate processing statistics
        processing_time = time.time() - start_time

        # Update accumulator
        self.accumulator["regions"] = region_results
        self.accumulator["region_histograms"] = self._fill_region_histograms(
            events, objects, region_masks
        )
        self.accumulator["region_cutflow"] = self._calculate_region_cutflow(region_masks)
        self.accumulator["region_validation"] = validation_results
        self.accumulator["metadata"] = {
            "n_events_processed": len(events),
            "n_regions": len(self.region_manager.regions),
            "processing_time": processing_time
        }

        logging.info(f"Analysis completed in {processing_time:.2f} seconds")

        return self.accumulator

    def _extract_objects_from_results(self, base_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract objects from base processor results.

        Args:
            base_results: Results from base processor

        Returns:
            Dictionary of objects
        """
        # This would extract objects from the base processor results
        # For now, return empty dict as placeholder
        return {}

    def _process_region(
        self,
        events: ak.Array,
        objects: Dict[str, Any],
        region_mask: ak.Array,
        region_name: str
    ) -> Dict[str, Any]:
        """
        Process a single region.

        Args:
            events: Awkward Array of events
            objects: Dictionary containing selected objects
            region_mask: Boolean mask for region
            region_name: Name of the region

        Returns:
            Region processing results
        """
        # Apply region mask
        region_events = events[region_mask]
        region_objects = {}

        # Handle empty regions
        if len(region_events) == 0:
            # Create empty objects for empty regions with proper structure
            for obj_name, obj_data in objects.items():
                if isinstance(obj_data, ak.Array):
                    # Create empty array with same structure (empty list of lists)
                    # Use ak.Array([]) but check if it's a list-type first
                    try:
                        # Try to create empty array with same layout
                        if hasattr(obj_data, 'type') and obj_data.type is not None:
                            # Create empty array with same type structure
                            region_objects[obj_name] = ak.Array([])
                        else:
                            region_objects[obj_name] = ak.Array([])
                    except Exception:
                        region_objects[obj_name] = ak.Array([])
                else:
                    region_objects[obj_name] = obj_data
        else:
            # Extract objects for non-empty regions
            for obj_name, obj_data in objects.items():
                if isinstance(obj_data, ak.Array):
                    region_objects[obj_name] = obj_data[region_mask]
                else:
                    region_objects[obj_name] = obj_data

        # Calculate region-specific variables
        region_variables = self._calculate_region_variables(region_events, region_objects)

        # Apply DNN if available
        dnn_scores = self._apply_dnn(region_events, region_objects)

        return {
            "n_events": len(region_events),
            "variables": region_variables,
            "dnn_scores": dnn_scores,
            "region_name": region_name
        }

    def _calculate_region_variables(
        self,
        events: ak.Array,
        objects: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate region-specific variables.

        Args:
            events: Awkward Array of events
            objects: Dictionary containing selected objects

        Returns:
            Dictionary of region variables
        """
        variables = {}

        # Handle empty events
        if len(events) == 0:
            return {}

        # MET
        variables["met"] = events["PFMET_pt"]

        # Jet multiplicity
        jets = objects.get("jets", ak.Array([]))
        bjets = objects.get("bjets", ak.Array([]))
        muons = objects.get("muons", ak.Array([]))
        electrons = objects.get("electrons", ak.Array([]))
        taus = objects.get("taus", ak.Array([]))

        # Check if objects are empty
        if len(ak.flatten(jets)) == 0:
            variables["n_jets"] = ak.zeros_like(events["event"], dtype=int)
        else:
            variables["n_jets"] = ak.num(jets, axis=1)

        if len(ak.flatten(bjets)) == 0:
            variables["n_bjets"] = ak.zeros_like(events["event"], dtype=int)
        else:
            variables["n_bjets"] = ak.num(bjets, axis=1)

        if len(ak.flatten(muons)) == 0:
            variables["n_muons"] = ak.zeros_like(events["event"], dtype=int)
        else:
            variables["n_muons"] = ak.num(muons, axis=1)

        if len(ak.flatten(electrons)) == 0:
            variables["n_electrons"] = ak.zeros_like(events["event"], dtype=int)
        else:
            variables["n_electrons"] = ak.num(electrons, axis=1)

        if len(ak.flatten(taus)) == 0:
            variables["n_taus"] = ak.zeros_like(events["event"], dtype=int)
        else:
            variables["n_taus"] = ak.num(taus, axis=1)

        # DeltaPhi between MET and jets
        jets = objects.get("jets", ak.Array([]))
        met_phi = events["PFMET_phi"]

        # Check if any events have jets
        n_jets_per_event = ak.num(jets, axis=1)
        has_jets = n_jets_per_event > 0

        if ak.any(has_jets):
            jet_phi = jets.phi
            # Calculate delta phi for each event, handling empty jet arrays
            delta_phi = ak.min(abs(jet_phi - met_phi), axis=1)
            # Fill NaN values with 0 for events with no jets
            delta_phi = ak.fill_none(delta_phi, 0.0)
            variables["delta_phi"] = delta_phi
        else:
            variables["delta_phi"] = ak.zeros_like(events["event"], dtype=float)

        return variables

    def _apply_dnn(
        self,
        events: ak.Array,
        objects: Dict[str, Any]
    ) -> Optional[ak.Array]:
        """
        Apply DNN to events (placeholder implementation).

        Args:
            events: Awkward Array of events
            objects: Dictionary containing selected objects

        Returns:
            DNN scores or None if DNN not available
        """
        # Placeholder for DNN application
        # In real implementation, this would load and apply the trained DNN model
        return None

    def _fill_region_histograms(
        self,
        events: ak.Array,
        objects: Dict[str, Any],
        region_masks: Dict[str, ak.Array]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fill histograms for all regions.

        Args:
            events: Awkward Array of events
            objects: Dictionary containing selected objects
            region_masks: Dictionary of region masks

        Returns:
            Dictionary of filled region histograms
        """
        filled_histograms = {}

        for region_name, region_mask in region_masks.items():
            # Check if region mask is valid and has any passing events
            n_passing = ak.sum(region_mask) if hasattr(region_mask, '__len__') else 0
            
            # Handle empty regions
            if n_passing == 0:
                # Create empty objects for empty regions
                region_objects = {}
                for obj_name, obj_data in objects.items():
                    if isinstance(obj_data, ak.Array):
                        region_objects[obj_name] = ak.Array([])
                    else:
                        region_objects[obj_name] = obj_data

                filled_histograms[region_name] = self.histogram_manager.define_histograms()
            else:
                # Apply region mask to events
                try:
                    region_events = events[region_mask]
                except Exception as e:
                    logging.warning(f"Error slicing events for region {region_name}: {e}, skipping")
                    filled_histograms[region_name] = self.histogram_manager.define_histograms()
                    continue

                # Extract objects for non-empty regions
                region_objects = {}
                for obj_name, obj_data in objects.items():
                    if isinstance(obj_data, ak.Array):
                        try:
                            region_objects[obj_name] = obj_data[region_mask]
                        except Exception as e:
                            logging.warning(f"Error slicing {obj_name} for region {region_name}: {e}, using empty array")
                            region_objects[obj_name] = ak.Array([])
                    else:
                        region_objects[obj_name] = obj_data

                # Fill histograms for this region
                if len(region_events) > 0:
                    filled_histograms[region_name] = self.histogram_manager.fill_histograms(
                        region_events, region_objects, ak.ones_like(region_events.event, dtype=float)
                    )
                else:
                    filled_histograms[region_name] = self.histogram_manager.define_histograms()

        return filled_histograms

    def _calculate_region_cutflow(self, region_masks: Dict[str, ak.Array]) -> Dict[str, Any]:
        """
        Calculate cutflow for all regions.

        Args:
            region_masks: Dictionary of region masks

        Returns:
            Cutflow results
        """
        cutflow = {
            "total_events": len(region_masks[list(region_masks.keys())[0]]) if region_masks else 0,
            "regions": {}
        }

        for region_name, mask in region_masks.items():
            n_events = ak.sum(mask)
            cutflow["regions"][region_name] = {
                "n_events": n_events,
                "fraction": float(n_events) / cutflow["total_events"] if cutflow["total_events"] > 0 else 0.0
            }

        return cutflow

    def get_region_summary(self) -> str:
        """
        Get formatted region summary.

        Returns:
            Formatted region summary string
        """
        summary = "Region Analysis Summary:\n"
        summary += "=" * 50 + "\n"

        # Region summary
        region_summary = self.region_manager.get_region_summary()
        summary += f"Number of regions: {region_summary['n_regions']}\n"
        summary += f"Signal regions: {self.region_manager.get_signal_regions()}\n"
        summary += f"Control regions: {self.region_manager.get_control_regions()}\n"
        summary += f"Validation regions: {self.region_manager.get_validation_regions()}\n\n"

        # Region details
        for region_name, region in self.region_manager.regions.items():
            summary += f"{region_name}:\n"
            summary += f"  Description: {region.description}\n"
            summary += f"  Cuts: {len(region.cuts)} cuts\n"
            summary += f"  Expected backgrounds: {region.expected_backgrounds}\n"
            summary += f"  Blind data: {region.blind_data}\n"
            summary += f"  Priority: {region.priority}\n"
            if region.transfer_factor_to_SR:
                summary += f"  Transfer factor: {region.transfer_factor_to_SR}\n"
            summary += "\n"

        return summary

    def get_region_validation_summary(self) -> str:
        """
        Get formatted region validation summary.

        Returns:
            Formatted validation summary string
        """
        validation = self.accumulator.get("region_validation", {})
        if not validation:
            return "No validation data available"

        summary = "Region Validation Summary:\n"
        summary += "=" * 50 + "\n"

        if validation.get("status") == "completed":
            # Region statistics
            summary += "Region Statistics:\n"
            for region_name, stats in validation.get("regions", {}).items():
                summary += f"  {region_name}: {stats['n_events']} events ({stats['fraction']:.2%})\n"

            # Overlaps
            if validation.get("overlaps"):
                summary += "\nRegion Overlaps:\n"
                for overlap_name, overlap_stats in validation["overlaps"].items():
                    summary += f"  {overlap_name}: {overlap_stats['n_overlap']} events ({overlap_stats['fraction']:.2%})\n"

            # Warnings
            if validation.get("warnings"):
                summary += "\nWarnings:\n"
                for warning in validation["warnings"]:
                    summary += f"  - {warning}\n"
        else:
            summary += f"Validation status: {validation.get('status', 'unknown')}\n"

        return summary

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

        # Convert region results to DataFrame format
        data = {}
        for region_name, region_data in self.accumulator.get("regions", {}).items():
            for var_name, var_data in region_data.get("variables", {}).items():
                col_name = f"{region_name}_{var_name}"
                if isinstance(var_data, ak.Array):
                    data[col_name] = ak.to_numpy(var_data)
                else:
                    data[col_name] = var_data

        df = pd.DataFrame(data)
        df.to_parquet(output_file)
        logging.info(f"Saved region results to {output_file}")

    def _save_root(self, output_file: str):
        """Save results as ROOT file."""
        try:
            import uproot
            with uproot.recreate(output_file) as f:
                # Save region histograms
                for region_name, histograms in self.accumulator.get("region_histograms", {}).items():
                    for hist_name, hist in histograms.items():
                        f[f"{region_name}_{hist_name}"] = hist

                # Save metadata
                f["metadata"] = self.accumulator.get("metadata", {})

            logging.info(f"Saved region results to {output_file}")
        except ImportError:
            logging.warning("uproot not available. Falling back to pickle.")
            self._save_pickle(output_file)

    def _save_pickle(self, output_file: str):
        """Save results as pickle file."""
        import pickle
        with open(output_file, 'wb') as f:
            pickle.dump(self.accumulator, f)
        logging.info(f"Saved region results to {output_file}")
