"""
Multi-region analyzer for DarkBottomLine framework.
"""

import awkward as ak
import numpy as np
import time
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
import json

from .processor import (
    DarkBottomLineProcessor,
    _build_event_weights_for_save,
    _event_weights_flat_columns,
)
from .objects import build_objects
from .regions import RegionManager
from .histograms import HistogramManager
from .selections import apply_selection
from .weights import WeightCalculator

# Try to import Coffea for processor wrapper
try:
    from coffea import processor
    COFFEA_AVAILABLE = True
except ImportError:
    COFFEA_AVAILABLE = False


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
            "metadata": {},
            "event_weights": {},
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

    def process(self, events: ak.Array, event_selection_output: Optional[str] = None) -> Dict[str, Any]:
        """
        Process events through all regions.

        Cuts are applied in series: (1) preselection (event selection), then
        (2) region cuts. Regions therefore only see events that passed preselection.

        Args:
            events: Awkward Array of events
            event_selection_output: If set, save preselected events to this path

        Returns:
            Analysis results with per-region histograms
        """
        start_time = time.time()

        # Build physics objects
        logging.info("Building physics objects...")
        objects = build_objects(events, self.base_processor.config)

        # Step 1 (series): Apply preselection first — regions get events after preselection
        logging.info("Applying preselection (event selection)...")
        try:
            selected_events, selected_objects, cutflow = apply_selection(
                events, objects, self.base_processor.config
            )
            events = selected_events
            objects = selected_objects
            logging.info(f"Events after preselection: {len(events)}")
        except Exception as e:
            logging.error(f"Preselection failed: {e}", exc_info=True)
            raise

        # Optionally save preselected events to file
        if event_selection_output:
            try:
                logging.info(f"Saving preselected events to {event_selection_output} ({len(events)} events)")
                self.base_processor._save_event_selection(event_selection_output, events, objects, max_events=self.base_processor.config.get("max_events"))
                import os
                if os.path.exists(event_selection_output):
                    file_size = os.path.getsize(event_selection_output)
                    logging.info(f"✓ Preselection saved to {event_selection_output} ({file_size} bytes)")
                else:
                    logging.error(f"✗ File {event_selection_output} was not created!")
            except Exception as e:
                logging.error(f"Failed to save preselection to {event_selection_output}: {e}", exc_info=True)

        # Step 2: Compute corrections and nominal total weight (for histograms + saving)
        logging.info("Computing corrections and event weights...")
        event_weights_nominal = None
        event_weights_save = {}
        try:
            corrections = self.base_processor.correction_manager.get_all_corrections(
                events, objects
            )
            weight_calculator = WeightCalculator(len(events))
            weight_calculator.add_generator_weight(events)
            weight_calculator.add_corrections(corrections)
            event_weights_nominal = weight_calculator.get_weight("central")
            event_weights_save = _build_event_weights_for_save(
                events, corrections, weight_calculator
            )
        except Exception as e:
            logging.warning(f"Weight calculation failed, using unit weights: {e}", exc_info=True)
            n_ev = len(events)
            event_weights_nominal = np.ones(n_ev, dtype=np.float64)
            event_weights_save = {
                "generator": np.ones(n_ev, dtype=np.float64),
                "pileup": np.ones(n_ev, dtype=np.float64),
                "weight_total_nominal": np.ones(n_ev, dtype=np.float64),
            }

        # Step 3 (series): Apply region cuts to preselected events only
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

        # Update accumulator (histograms filled with nominal total weight; all systematics saved)
        self.accumulator["regions"] = region_results
        self.accumulator["region_histograms"] = self._fill_region_histograms(
            events, objects, region_masks, event_weights_nominal
        )
        self.accumulator["region_cutflow"] = self._calculate_region_cutflow(region_masks)
        self.accumulator["region_validation"] = validation_results
        self.accumulator["event_weights"] = event_weights_save
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
        variables["met"] = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]

        # Jet multiplicity; use tight pt>30 leptons for region-consistent variables
        jets = objects.get("jets", ak.Array([]))
        bjets = objects.get("bjets", ak.Array([]))
        muons = objects.get("tight_muons_pt30", ak.Array([]))
        electrons = objects.get("tight_electrons_pt30", ak.Array([]))
        taus = objects.get("tight_taus_pt30", ak.Array([]))

        # Check if objects are empty
        # Safe per-event count: use axis=1 for jagged arrays; depth-1 (sliced region) can raise
        def _safe_num_jagged(arr, n_ev: int):
            if arr is None or (isinstance(arr, ak.Array) and len(arr) == 0):
                return np.zeros(n_ev, dtype=np.int64)
            try:
                return np.asarray(ak.num(arr, axis=1))
            except (Exception, BaseException):
                return np.zeros(n_ev, dtype=np.int64)

        n_ev = len(events) # Total events in this region

        variables["n_jets"] = _safe_num_jagged(objects.get("jets"), n_ev)
        variables["n_bjets"] = _safe_num_jagged(objects.get("bjets"), n_ev)
        variables["n_muons"] = _safe_num_jagged(objects.get("tight_muons_pt30"), n_ev)
        variables["n_electrons"] = _safe_num_jagged(objects.get("tight_electrons_pt30"), n_ev)
        variables["n_taus"] = _safe_num_jagged(objects.get("tight_taus_pt30"), n_ev)

        # DeltaPhi between MET and jets (use safe count; avoid axis=1 on depth-1 arrays)
        jets = objects.get("jets", ak.Array([]))
        met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]
        n_jets_per_event = _safe_num_jagged(jets, n_ev)
        has_jets = np.any(n_jets_per_event > 0)

        if has_jets:
            try:
                jet_phi = jets.phi
                delta_phi = ak.min(ak.abs(jet_phi - met_phi), axis=1)
                delta_phi = ak.fill_none(delta_phi, 0.0)
                variables["delta_phi"] = delta_phi
            except (Exception, BaseException):
                variables["delta_phi"] = np.zeros(n_ev, dtype=np.float64)
        else:
            variables["delta_phi"] = np.zeros(n_ev, dtype=np.float64)

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
        region_masks: Dict[str, ak.Array],
        event_weights_nominal: Optional[Union[ak.Array, np.ndarray]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fill histograms for all regions with nominal total weight.

        Args:
            events: Awkward Array of events
            objects: Dictionary containing selected objects
            region_masks: Dictionary of region masks
            event_weights_nominal: Per-event nominal total weight (optional); if None, use ones.

        Returns:
            Dictionary of filled region histograms
        """
        filled_histograms = {}
        n_ev = len(events)
        # Convert to numpy once for slicing per region
        if event_weights_nominal is not None:
            try:
                w_full = np.asarray(ak.to_numpy(event_weights_nominal))
            except Exception:
                w_full = np.ones(n_ev, dtype=np.float64)
        else:
            w_full = np.ones(n_ev, dtype=np.float64)

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

                # Fill histograms with nominal total weight (sliced for this region)
                if len(region_events) > 0:
                    n_r = len(region_events)
                    try:
                        mask_np = np.asarray(ak.to_numpy(region_mask))
                        w = w_full[mask_np].astype(np.float64)
                    except (Exception, BaseException):
                        w = np.ones(n_r, dtype=np.float64)
                    filled_histograms[region_name] = self.histogram_manager.fill_histograms(
                        region_events, region_objects, w
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
        """Save results as Parquet file (region variables + per-event weights)."""
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

        # Per-event weights (central/up/down per systematic)
        event_weights = self.accumulator.get("event_weights", {})
        if event_weights:
            flat = _event_weights_flat_columns(event_weights)
            for k, v in flat.items():
                data[k] = v

        df = pd.DataFrame(data)
        df.to_parquet(output_file)
        logging.info(f"Saved region results to {output_file}")

    def _save_root(self, output_file: str):
        """Save results as ROOT file (histograms + per-event weight tree)."""
        try:
            import uproot
            with uproot.recreate(output_file) as f:
                # Save region histograms
                for region_name, histograms in self.accumulator.get("region_histograms", {}).items():
                    for hist_name, hist in histograms.items():
                        f[f"{region_name}_{hist_name}"] = hist

                # Save metadata as a JSON string
                metadata_str = json.dumps(self.accumulator.get("metadata", {}))
                f["metadata"] = metadata_str

                # Per-event weights as TTree
                event_weights = self.accumulator.get("event_weights", {})
                if event_weights:
                    flat = _event_weights_flat_columns(event_weights)
                    if flat:
                        f["event_weights"] = flat

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


# Coffea processor wrapper for analyzer compatibility
if COFFEA_AVAILABLE:
    class DarkBottomLineAnalyzerCoffeaProcessor(processor.ProcessorABC):
        """
        Coffea-compatible processor wrapper for multi-region analyzer.
        """

        def __init__(self, config: Dict[str, Any], regions_config_path: str, event_selection_output: Optional[str] = None):
            self.config = config
            self.regions_config_path = regions_config_path
            self.event_selection_output = event_selection_output
            self.analyzer = DarkBottomLineAnalyzer(config, regions_config_path)
            # Initialize accumulator for Coffea
            self.accumulator = processor.dict_accumulator({
                "regions": processor.dict_accumulator({}),
                "region_histograms": processor.dict_accumulator({}),
                "region_cutflow": processor.dict_accumulator({}),
                "region_validation": processor.dict_accumulator({}),
                "metadata": processor.dict_accumulator({}),
                "event_weights": processor.dict_accumulator({}),
            })
            # Store selected events/objects for event_selection_output
            # Use file-based approach for cross-worker compatibility
            if event_selection_output:
                import tempfile
                import os
                import uuid
                # Create a temp directory for chunk files (use a unique ID to avoid conflicts)
                unique_id = str(uuid.uuid4())[:8]
                self._temp_dir = tempfile.mkdtemp(prefix=f"dbl_event_selection_{unique_id}_")
                # Store chunk file list in accumulator (for cross-worker merging)
                # Use a dict accumulator with file paths as keys (for easy merging across workers)
                # Note: each worker may have its own temp_dir, but we'll handle that in postprocess
                self.accumulator["_event_selection_chunk_files"] = processor.dict_accumulator({})

        def process(self, events: ak.Array) -> Dict[str, Any]:
            """Process events using the analyzer."""
            # Don't save event_selection_output per chunk, accumulate instead
            result = self.analyzer.process(events, event_selection_output=None)

            # If event_selection_output is requested, collect selected events from this chunk
            if self.event_selection_output:
                try:
                    import os
                    import pickle
                    import awkward as ak
                    logging.info(f"Collecting selected events for event_selection_output from chunk ({len(events)} events)")
                    from .selections import apply_selection
                    from .objects import build_objects
                    # Apply event-level selection to get selected events
                    objects = build_objects(events, self.config)
                    selected_events, selected_objects, _ = apply_selection(
                        events, objects, self.config
                    )
                    logging.info(f"Chunk: {len(selected_events)}/{len(events)} events passed selection")

                    # Save this chunk to a temporary file (for cross-worker compatibility)
                    # Use unique filename to avoid conflicts across workers
                    import time
                    import uuid
                    chunk_id = f"{int(time.time() * 1000000)}_{uuid.uuid4().hex[:8]}"
                    chunk_file = os.path.join(self._temp_dir, f"chunk_{chunk_id}.pkl")
                    with open(chunk_file, 'wb') as f:
                        pickle.dump({"events": selected_events, "objects": selected_objects}, f, protocol=pickle.HIGHEST_PROTOCOL)

                    # Add to accumulator dict (this will be merged across workers)
                    # Use file path as key with dummy value for easy merging
                    self.accumulator["_event_selection_chunk_files"][chunk_file] = True
                    logging.info(f"Saved chunk to {chunk_file}, added to accumulator")
                except Exception as e:
                    logging.warning(f"Failed to collect selected events for event_selection_output: {e}", exc_info=True)
            # Update accumulator with results
            if "regions" in result:
                for key, value in result["regions"].items():
                    if key not in self.accumulator["regions"]:
                        self.accumulator["regions"][key] = value
                    else:
                        # Merge if needed
                        if isinstance(self.accumulator["regions"][key], dict) and isinstance(value, dict):
                            self.accumulator["regions"][key].update(value)
            if "region_histograms" in result:
                for region_name, histograms in result["region_histograms"].items():
                    if region_name not in self.accumulator["region_histograms"]:
                        self.accumulator["region_histograms"][region_name] = processor.dict_accumulator({})
                    for hist_name, hist in histograms.items():
                        if hist_name in self.accumulator["region_histograms"][region_name]:
                            # Add histograms if they support it
                            if hasattr(self.accumulator["region_histograms"][region_name][hist_name], 'add'):
                                self.accumulator["region_histograms"][region_name][hist_name].add(hist)
                            else:
                                self.accumulator["region_histograms"][region_name][hist_name] = hist
                        else:
                            self.accumulator["region_histograms"][region_name][hist_name] = hist
            if "region_cutflow" in result:
                for key, value in result["region_cutflow"].items():
                    if key not in self.accumulator["region_cutflow"]:
                        # Wrap plain dict in dict_accumulator for proper Coffea merging
                        if isinstance(value, dict):
                            self.accumulator["region_cutflow"][key] = processor.dict_accumulator(value)
                        else:
                            self.accumulator["region_cutflow"][key] = value
                    else:
                        # Merge cutflow dictionaries - handle both dict_accumulator and plain dict
                        if isinstance(value, dict):
                            # If accumulator value is also a dict_accumulator, merge properly
                            if hasattr(self.accumulator["region_cutflow"][key], 'add'):
                                # It's a dict_accumulator, create a temporary one and add
                                temp_acc = processor.dict_accumulator(value)
                                self.accumulator["region_cutflow"][key].add(temp_acc)
                            else:
                                # It's a plain dict, merge manually
                                for k, v in value.items():
                                    if k not in self.accumulator["region_cutflow"][key]:
                                        self.accumulator["region_cutflow"][key][k] = v
                                    else:
                                        self.accumulator["region_cutflow"][key][k] = self.accumulator["region_cutflow"][key][k] + v
            if "region_validation" in result:
                # Handle region_validation - wrap in dict_accumulator for proper merging
                validation = result["region_validation"]
                if isinstance(validation, dict):
                    # Merge validation results
                    for key, value in validation.items():
                        if key not in self.accumulator["region_validation"]:
                            # Wrap plain dict/values in dict_accumulator if needed
                            if isinstance(value, dict):
                                self.accumulator["region_validation"][key] = processor.dict_accumulator(value)
                            else:
                                self.accumulator["region_validation"][key] = value
                        else:
                            # Merge validation dictionaries
                            if isinstance(value, dict):
                                if hasattr(self.accumulator["region_validation"][key], 'add'):
                                    temp_acc = processor.dict_accumulator(value)
                                    self.accumulator["region_validation"][key].add(temp_acc)
                                else:
                                    # Plain dict merge
                                    if isinstance(self.accumulator["region_validation"][key], dict):
                                        self.accumulator["region_validation"][key].update(value)
            if "metadata" in result:
                self.accumulator["metadata"].update(result.get("metadata", {}))
            # Merge event_weights (concatenate arrays across chunks)
            if "event_weights" in result and result["event_weights"]:
                self._merge_event_weights_into_accumulator(result["event_weights"])
            return self.accumulator

        def _merge_event_weights_into_accumulator(self, new_weights: Dict[str, Any]):
            """Merge new per-event weights into accumulator (concatenate arrays)."""
            acc = self.accumulator["event_weights"]
            if isinstance(acc, dict) and not acc:
                # First chunk: store as plain dict (numpy arrays)
                merged = {}
                for k, v in new_weights.items():
                    if isinstance(v, dict):
                        merged[k] = {kk: np.asarray(vv) for kk, vv in v.items() if isinstance(vv, np.ndarray)}
                    elif isinstance(v, np.ndarray):
                        merged[k] = v
                    else:
                        merged[k] = np.asarray(v)
                self.accumulator["event_weights"] = merged
                return
            # Subsequent chunks: concatenate
            merged = self.accumulator["event_weights"] if isinstance(self.accumulator["event_weights"], dict) else {}
            for k, v in new_weights.items():
                if isinstance(v, dict):
                    if k not in merged:
                        merged[k] = {}
                    for kk, vv in v.items():
                        if isinstance(vv, np.ndarray):
                            merged[k][kk] = np.concatenate([merged[k][kk], vv]) if kk in merged[k] else vv
                elif isinstance(v, np.ndarray):
                    merged[k] = np.concatenate([merged[k], v]) if k in merged else v
            self.accumulator["event_weights"] = merged

        def postprocess(self, accumulator: Dict[str, Any]) -> Dict[str, Any]:
            """Post-process results."""
            import os
            import pickle
            import awkward as ak

            # Get chunk files from accumulator (merged across all workers)
            # File paths are stored as keys in the dict accumulator
            chunk_files_acc = accumulator.get("_event_selection_chunk_files", {})
            if isinstance(chunk_files_acc, dict):
                chunk_files = list(chunk_files_acc.keys())
            else:
                chunk_files = []

            logging.info(f"postprocess called: event_selection_output={self.event_selection_output}, chunk_files={len(chunk_files)}")

            # Save accumulated event selection if requested
            if self.event_selection_output and chunk_files:
                try:
                    # Load all chunks from files
                    all_selected_events_list = []
                    all_selected_objects_list = []

                    for chunk_file in chunk_files:
                        if os.path.exists(chunk_file):
                            try:
                                with open(chunk_file, 'rb') as f:
                                    chunk_data = pickle.load(f)
                                    events = chunk_data.get("events")
                                    objects = chunk_data.get("objects")
                                    if events is not None and len(events) > 0:
                                        all_selected_events_list.append(events)
                                        all_selected_objects_list.append(objects)
                            except Exception as e:
                                logging.warning(f"Failed to load chunk file {chunk_file}: {e}")

                    if all_selected_events_list:
                        # Concatenate all selected events and objects from all chunks
                        all_selected_events = ak.concatenate(all_selected_events_list)

                        # Merge selected objects dictionaries
                        all_selected_objects = {}
                        if all_selected_objects_list:
                            # Get all keys from all chunks
                            all_keys = set()
                            for obj_dict in all_selected_objects_list:
                                all_keys.update(obj_dict.keys())
                            # Concatenate arrays for each key
                            for key in all_keys:
                                arrays_to_concat = []
                                for obj_dict in all_selected_objects_list:
                                    if key in obj_dict and len(obj_dict[key]) > 0:
                                        arrays_to_concat.append(obj_dict[key])
                                if arrays_to_concat:
                                    all_selected_objects[key] = ak.concatenate(arrays_to_concat)

                        # Save using base processor helper
                        logging.info(f"Saving accumulated event-level selection to {self.event_selection_output}")
                        self.analyzer.base_processor._save_event_selection(
                            self.event_selection_output, all_selected_events, all_selected_objects, max_events=self.config.get("max_events")
                        )
                        # Verify file was created
                        if os.path.exists(self.event_selection_output):
                            file_size = os.path.getsize(self.event_selection_output)
                            logging.info(f"✓ Saved accumulated event-level selection from {len(all_selected_events_list)} chunks ({len(all_selected_events)} events) to {self.event_selection_output} ({file_size} bytes)")
                        else:
                            logging.error(f"✗ File {self.event_selection_output} was not created!")
                    else:
                        logging.warning(f"No selected events found in {len(chunk_files)} chunk files, skipping event_selection_output save")

                    # Clean up temporary files and directories
                    # Collect all unique temp directories from chunk file paths
                    temp_dirs = set()
                    for chunk_file in chunk_files:
                        if os.path.exists(chunk_file):
                            temp_dirs.add(os.path.dirname(chunk_file))

                    # Remove chunk files and temp directories
                    for chunk_file in chunk_files:
                        try:
                            if os.path.exists(chunk_file):
                                os.remove(chunk_file)
                        except Exception as e:
                            logging.warning(f"Failed to remove chunk file {chunk_file}: {e}")

                    for temp_dir in temp_dirs:
                        try:
                            if os.path.exists(temp_dir) and os.path.isdir(temp_dir):
                                os.rmdir(temp_dir)
                                logging.debug(f"Removed temp directory {temp_dir}")
                        except Exception as e:
                            logging.warning(f"Failed to remove temp directory {temp_dir}: {e}")

                except Exception as e:
                    logging.error(f"Failed to save accumulated event selection to {self.event_selection_output}: {e}", exc_info=True)

            return accumulator
