"""
Weight calculation and combination using Coffea's Weights class.
"""

import awkward as ak
import numpy as np
from typing import Dict, Any, Optional, Union
import logging

try:
    from coffea.analysis_tools import Weights
    COFFEA_WEIGHTS_AVAILABLE = True
except ImportError:
    COFFEA_WEIGHTS_AVAILABLE = False
    logging.warning("Coffea Weights not available. Using fallback implementation.")


class WeightCalculator:
    """
    Calculator for combining various weights using Coffea's Weights class.
    """

    def __init__(self, size: int, storeIndividual: bool = True):
        """
        Initialize weight calculator.

        Args:
            size: Size of the event array
            storeIndividual: Whether to store individual weights
        """
        self.size = size
        self.storeIndividual = storeIndividual

        if COFFEA_WEIGHTS_AVAILABLE:
            self.weights = Weights(size, storeIndividual=storeIndividual)
        else:
            # Fallback implementation
            self.weights = None
            self.weight_dict = {}
            self.weight_variations = {}

    def add_weight(
        self,
        name: str,
        weight: Union[ak.Array, np.ndarray, float],
        weightUp: Optional[Union[ak.Array, np.ndarray, float]] = None,
        weightDown: Optional[Union[ak.Array, np.ndarray, float]] = None
    ):
        """
        Add a weight to the calculator.

        Args:
            name: Name of the weight
            weight: Central weight values
            weightUp: Up variation (optional)
            weightDown: Down variation (optional)
        """
        if COFFEA_WEIGHTS_AVAILABLE:
            if weightUp is not None and weightDown is not None:
                self.weights.add(name, weight, weightUp, weightDown)
            else:
                self.weights.add(name, weight)
        else:
            # Fallback implementation
            self.weight_dict[name] = weight
            if weightUp is not None:
                self.weight_variations[f"{name}_up"] = weightUp
            if weightDown is not None:
                self.weight_variations[f"{name}_down"] = weightDown

    def add_generator_weight(self, events: ak.Array):
        """
        Add generator weight from events.

        Args:
            events: Awkward Array of events
        """
        if hasattr(events, 'genWeight'):
            gen_weight = events.genWeight
            # Handle negative weights by taking absolute value
            gen_weight = np.abs(gen_weight)
            self.add_weight("generator", gen_weight)
        else:
            logging.warning("Generator weights not found in events")

    def add_pileup_weight(self, pileup_weight: Union[ak.Array, np.ndarray]):
        """
        Add pileup reweighting.

        Args:
            pileup_weight: Pileup weight values
        """
        self.add_weight("pileup", pileup_weight)

    def add_muon_sf(self, muon_sf: Union[ak.Array, np.ndarray]):
        """
        Add muon scale factors.

        Args:
            muon_sf: Muon scale factor values
        """
        self.add_weight("muon_sf", muon_sf)

    def add_electron_sf(self, electron_sf: Union[ak.Array, np.ndarray]):
        """
        Add electron scale factors.

        Args:
            electron_sf: Electron scale factor values
        """
        self.add_weight("electron_sf", electron_sf)

    def add_btag_sf(self, btag_sf: Union[ak.Array, np.ndarray]):
        """
        Add b-tagging scale factors.

        Args:
            btag_sf: B-tagging scale factor values
        """
        self.add_weight("btag_sf", btag_sf)

    def add_corrections(
        self,
        corrections: Dict[str, Union[ak.Array, np.ndarray, Dict[str, Union[ak.Array, np.ndarray]]]]
    ):
        """
        Add multiple corrections at once.

        Args:
            corrections: Dictionary of correction names and values. Each value is either
                a per-event array (e.g. pileup) or a dict {"central", "up", "down"} so
                the combined weight uses central only and variations use up/down.
        """
        for name, weight in corrections.items():
            if isinstance(weight, dict) and "central" in weight:
                self.add_weight(
                    name,
                    weight["central"],
                    weightUp=weight.get("up"),
                    weightDown=weight.get("down"),
                )
            else:
                self.add_weight(name, weight)

    def get_weight(self, variation: str = "central") -> Union[ak.Array, np.ndarray]:
        """
        Get the combined weight for a given variation.

        Final event weight = product of all per-event weights (generator, pileup,
        weight_muon_id, weight_electron_id, weight_btag, etc.). Each component is
        the product over objects (e.g. weight_btag = product of all b-tag SFs).

        Args:
            variation: Weight variation ("central", "up", "down")

        Returns:
            Combined weight array (one value per event)
        """
        if COFFEA_WEIGHTS_AVAILABLE:
            if variation == "central":
                return self.weights.weight()
            elif variation == "up":
                return self.weights.weight("up")
            elif variation == "down":
                return self.weights.weight("down")
            else:
                return self.weights.weight(variation)
        else:
            # Fallback implementation
            if variation == "central":
                # Multiply all central weights
                result = np.ones(self.size)
                for name, weight in self.weight_dict.items():
                    if isinstance(weight, (ak.Array, np.ndarray)):
                        result *= ak.to_numpy(weight)
                    else:
                        result *= weight
                return result
            else:
                # Handle variations
                var_name = f"_{variation}"
                result = np.ones(self.size)
                for name, weight in self.weight_dict.items():
                    if isinstance(weight, (ak.Array, np.ndarray)):
                        result *= ak.to_numpy(weight)
                    else:
                        result *= weight

                # Apply variations
                for name, weight in self.weight_variations.items():
                    if name.endswith(var_name):
                        base_name = name.replace(var_name, "")
                        if base_name in self.weight_dict:
                            if isinstance(weight, (ak.Array, np.ndarray)):
                                result *= ak.to_numpy(weight)
                            else:
                                result *= weight
                return result

    def get_individual_weights(self) -> Dict[str, Union[ak.Array, np.ndarray]]:
        """
        Get individual weights.

        Returns:
            Dictionary of individual weights
        """
        if COFFEA_WEIGHTS_AVAILABLE:
            return self.weights.weight()
        else:
            return self.weight_dict

    def get_weight_names(self) -> list:
        """
        Get list of weight names.

        Returns:
            List of weight names
        """
        if COFFEA_WEIGHTS_AVAILABLE:
            return self.weights.weight()
        else:
            return list(self.weight_dict.keys())

    def get_variations(self) -> list:
        """
        Get list of available weight variations.

        Returns:
            List of variation names
        """
        if COFFEA_WEIGHTS_AVAILABLE:
            return self.weights.variations
        else:
            variations = ["central"]
            for name in self.weight_variations.keys():
                if name.endswith("_up"):
                    variations.append("up")
                elif name.endswith("_down"):
                    variations.append("down")
            return list(set(variations))

    def add_systematic_variations(
        self,
        corrections: Dict[str, Dict[str, Union[ak.Array, np.ndarray]]]
    ):
        """
        Add systematic variations for corrections.

        Args:
            corrections: Dictionary with systematic variations
                Format: {"correction_name": {"central": values, "up": values, "down": values}}
        """
        for name, variations in corrections.items():
            if "central" in variations:
                self.add_weight(
                    name,
                    variations["central"],
                    variations.get("up"),
                    variations.get("down")
                )

    def calculate_event_weights(
        self,
        events: ak.Array,
        objects: Dict[str, Any],
        corrections: Dict[str, Union[ak.Array, np.ndarray]]
    ) -> Union[ak.Array, np.ndarray]:
        """
        Calculate final event weights combining all corrections.

        Args:
            events: Awkward Array of events
            objects: Dictionary containing selected objects
            corrections: Dictionary of correction weights

        Returns:
            Final event weights
        """
        # Add generator weight
        self.add_generator_weight(events)

        # Add corrections
        self.add_corrections(corrections)

        # Get final weight
        return self.get_weight("central")

    def get_weight_statistics(self) -> Dict[str, float]:
        """
        Get statistics about the weights.

        Returns:
            Dictionary with weight statistics
        """
        weights = self.get_weight("central")

        if isinstance(weights, (ak.Array, np.ndarray)):
            weights = ak.to_numpy(weights)

        # Handle empty arrays
        if len(weights) == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "sum": 0.0,
            }

        return {
            "mean": float(np.mean(weights)),
            "std": float(np.std(weights)),
            "min": float(np.min(weights)),
            "max": float(np.max(weights)),
            "sum": float(np.sum(weights)),
        }
