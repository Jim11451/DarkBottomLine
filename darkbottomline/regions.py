"""
Region definition and management for DarkBottomLine analysis.
"""

import awkward as ak
import numpy as np
import yaml
from typing import Dict, Any, List, Optional, Union
import logging


class Region:
    """
    Represents a single analysis region with its cuts and properties.
    """

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize region from configuration.

        Args:
            name: Region name
            config: Region configuration dictionary
        """
        self.name = name
        self.description = config.get("description", "")
        self.cuts = config.get("cuts", {})
        self.expected_backgrounds = config.get("expected_backgrounds", [])
        self.blind_data = config.get("blind_data", False)
        self.priority = config.get("priority", 1)
        self.transfer_factor_to_SR = config.get("transfer_factor_to_SR", None)

        # Parse cuts into evaluable expressions
        self.parsed_cuts = self._parse_cuts()

    def _parse_cuts(self) -> Dict[str, Dict[str, Any]]:
        """
        Parse cut strings into evaluable expressions.

        Returns:
            Dictionary of parsed cuts
        """
        parsed = {}

        for var, cut_str in self.cuts.items():
            if ">=" in cut_str:
                value = float(cut_str.replace(">=", ""))
                parsed[var] = {"operator": ">=", "value": value}
            elif "<=" in cut_str:
                value = float(cut_str.replace("<=", ""))
                parsed[var] = {"operator": "<=", "value": value}
            elif ">" in cut_str:
                value = float(cut_str.replace(">", ""))
                parsed[var] = {"operator": ">", "value": value}
            elif "<" in cut_str:
                value = float(cut_str.replace("<", ""))
                parsed[var] = {"operator": "<", "value": value}
            elif "==" in cut_str:
                value = float(cut_str.replace("==", ""))
                parsed[var] = {"operator": "==", "value": value}
            elif "!=" in cut_str:
                value = float(cut_str.replace("!=", ""))
                parsed[var] = {"operator": "!=", "value": value}
            else:
                try:
                    value = float(cut_str)
                    parsed[var] = {"operator": "==", "value": value}
                except ValueError:
                    logging.warning(f"Could not parse cut: {var} {cut_str}")

        return parsed

    def apply_cuts(self, events: ak.Array, objects: Dict[str, Any]) -> ak.Array:
        """
        Apply region cuts to events.
        """
        mask = ak.ones_like(events.event, dtype=bool)

        for var, cut_info in self.parsed_cuts.items():
            operator = cut_info["operator"]
            value = cut_info["value"]

            # Get variable value
            var_value = self._get_variable_value(events, objects, var)

            if var_value is None:
                logging.warning(f"Variable {var} not found, skipping cut")
                continue

            # Apply cut
            if operator == ">":
                cut_mask = var_value > value
            elif operator == ">=":
                cut_mask = var_value >= value
            elif operator == "<":
                cut_mask = var_value < value
            elif operator == "<=":
                cut_mask = var_value <= value
            elif operator == "==":
                cut_mask = var_value == value
            elif operator == "!=":
                cut_mask = var_value != value
            else:
                logging.warning(f"Unknown operator: {operator}")
                continue

            mask = mask & ak.fill_none(cut_mask, False)

        return mask

    def _get_variable_value(self, events: ak.Array, objects: Dict[str, Any], var: str) -> Optional[ak.Array]:
        """
        Get variable value from events or objects.
        """
        # Special variables
        if var == "MET":
            try:
                if "PFMET_pt" in events.fields:
                    return ak.fill_none(events["PFMET_pt"], 0.0)
                else:
                    return ak.fill_none(events["MET_pt"], 0.0)
            except Exception:
                if "PFMET_pt" in events.fields:
                    return events["PFMET_pt"]
                else:
                    return events["MET_pt"]
        if var == "Nbjets":
            return ak.num(objects.get("bjets", ak.Array([])), axis=1)
        if var == "Njets" or var == "NjetsMin":
            return ak.num(objects.get("jets", ak.Array([])), axis=1)
        if var == "Jet1Pt":
            jets = objects.get("jets", ak.Array([]))
            if len(ak.flatten(jets)) == 0:
                return ak.zeros_like(events.event, dtype=float)
            try:
                return ak.fill_none(ak.max(jets.pt, axis=1), 0.0)
            except Exception:
                return ak.max(jets.pt, axis=1)
        if var == "Nleptons":
            n_muons = ak.num(objects.get("muons", ak.Array([])), axis=1)
            n_electrons = ak.num(objects.get("electrons", ak.Array([])), axis=1)
            n_taus = ak.num(objects.get("taus", ak.Array([])), axis=1)
            return n_muons + n_electrons + n_taus
        if var == "Nmuons":
            return ak.num(objects.get("muons", ak.Array([])), axis=1)
        if var == "Nelectrons":
            return ak.num(objects.get("electrons", ak.Array([])), axis=1)
        if var == "Ntaus":
            return ak.num(objects.get("taus", ak.Array([])), axis=1)
        if var == "NAdditionalJets":
            n_jets = ak.num(objects.get("jets", ak.Array([])), axis=1)
            n_bjets = ak.num(objects.get("bjets", ak.Array([])), axis=1)
            return n_jets - n_bjets
        if var == "Recoil":
            # Recoil = | -( pTmiss + sum pT(leptons) ) |
            met_pt = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]
            met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]

            # Get lepton momenta
            muons = objects.get("muons", ak.Array([]))
            electrons = objects.get("electrons", ak.Array([]))

            # Calculate lepton pT sum
            lep_pt_sum = ak.zeros_like(met_pt)
            if len(ak.flatten(muons)) > 0:
                lep_pt_sum = lep_pt_sum + ak.sum(muons.pt, axis=1)
            if len(ak.flatten(electrons)) > 0:
                lep_pt_sum = lep_pt_sum + ak.sum(electrons.pt, axis=1)

            # Calculate Recoil magnitude
            recoil_px = -(met_pt * np.cos(met_phi) + lep_pt_sum)
            recoil_py = -(met_pt * np.sin(met_phi))
            recoil = np.sqrt(recoil_px**2 + recoil_py**2)

            return ak.fill_none(recoil, 0.0)
        if var == "MT":
            # Transverse mass = sqrt(2 * pT_lepton * MET * (1 - cos(Δφ)))
            met_pt = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]
            met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]

            muons = objects.get("muons", ak.Array([]))
            electrons = objects.get("electrons", ak.Array([]))

            mt = ak.zeros_like(met_pt)

            # Muon MT
            has_muons = ak.num(muons) > 0
            leading_muon = ak.firsts(muons[ak.argsort(muons.pt, ascending=False, axis=1)])
            muon_pt = leading_muon.pt
            muon_phi = leading_muon.phi
            delta_phi_mu = abs(muon_phi - met_phi)
            delta_phi_mu = ak.where(delta_phi_mu > np.pi, 2 * np.pi - delta_phi_mu, delta_phi_mu)
            mt_mu = np.sqrt(2 * muon_pt * met_pt * (1 - np.cos(delta_phi_mu)))
            
            mt = ak.where(has_muons, mt_mu, mt)

            # Electron MT for events without muons
            has_electrons = ak.num(electrons) > 0
            leading_electron = ak.firsts(electrons[ak.argsort(electrons.pt, ascending=False, axis=1)])
            ele_pt = leading_electron.pt
            ele_phi = leading_electron.phi
            delta_phi_el = abs(ele_phi - met_phi)
            delta_phi_el = ak.where(delta_phi_el > np.pi, 2 * np.pi - delta_phi_el, delta_phi_el)
            mt_el = np.sqrt(2 * ele_pt * met_pt * (1 - np.cos(delta_phi_el)))

            mt = ak.where(~has_muons & has_electrons, mt_el, mt)
            
            return ak.fill_none(mt, 0.0)
        if var in ("Mll", "MllMin", "MllMax"):
            # Simplified dilepton invariant mass calculation
            muons = objects.get("muons", ak.Array([]))
            electrons = objects.get("electrons", ak.Array([]))

            mll = ak.zeros_like(events["event"], dtype=float)

            # For now, return a placeholder value
            # TODO: Implement proper dilepton mass calculation
            return ak.fill_none(mll, 0.0)
        if var == "DeltaPhi":
            jets = objects.get("jets", ak.Array([]))
            met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]

            # Check if jets array is empty or has no structure
            if len(ak.flatten(jets)) == 0 or len(jets) == 0:
                return ak.zeros_like(events["event"], dtype=float)

            try:
                n_jets_per_event = ak.num(jets, axis=1)
                has_jets = n_jets_per_event > 0
                if ak.any(has_jets):
                    jet_phi = jets.phi
                    delta_phi = ak.min(np.abs(jet_phi - met_phi), axis=1)
                    return ak.fill_none(delta_phi, 0.0)
            except Exception:
                pass

            return ak.zeros_like(events["event"], dtype=float)
        if var == "LeptonPt":
            muons = objects.get("muons", ak.Array([]))
            electrons = objects.get("electrons", ak.Array([]))
            taus = objects.get("taus", ak.Array([]))
            all_leptons = ak.concatenate([muons, electrons, taus], axis=1)
            if len(ak.flatten(all_leptons)) == 0:
                return ak.zeros_like(events.event, dtype=float)
            try:
                return ak.fill_none(ak.max(all_leptons.pt, axis=1), 0.0)
            except Exception:
                return ak.max(all_leptons.pt, axis=1)
        if var == "metQuality":
            # MET Quality = (pfMET - caloMET) / Recoil
            pf_met = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]
            calo_met = events.get("CaloMET_pt", pf_met)  # Fallback to pfMET if caloMET not available
            recoil = self._get_variable_value(events, objects, "Recoil")

            # Avoid division by zero
            met_quality = ak.where(recoil > 0, (pf_met - calo_met) / recoil, 0.0)
            return ak.fill_none(met_quality, 0.0)

        # Direct from events/objects
        if hasattr(events, var):
            try:
                return ak.fill_none(getattr(events, var), 0)
            except Exception:
                return getattr(events, var)
        for obj_name, obj_data in objects.items():
            if hasattr(obj_data, var):
                try:
                    return ak.fill_none(getattr(obj_data, var), 0)
                except Exception:
                    return getattr(obj_data, var)
        return ak.zeros_like(events.event, dtype=float)


class RegionManager:
    """
    Manages multiple analysis regions and their application.
    """

    def __init__(self, config_path: str):
        """
        Initialize region manager from configuration file.

        Args:
            config_path: Path to regions configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.regions = {}
        self.settings = self.config.get("settings", {})
        self.validation = self.config.get("validation", {})

        # Create region objects
        for region_name, region_config in self.config.get("regions", {}).items():
            self.regions[region_name] = Region(region_name, region_config)

        logging.info(f"Loaded {len(self.regions)} regions: {list(self.regions.keys())}")

    def get_region(self, name: str) -> Optional[Region]:
        return self.regions.get(name)

    def get_all_regions(self) -> Dict[str, Region]:
        return self.regions

    def apply_regions(self, events: ak.Array, objects: Dict[str, Any]) -> Dict[str, ak.Array]:
        region_masks = {}
        for region_name, region in self.regions.items():
            mask = region.apply_cuts(events, objects)
            region_masks[region_name] = ak.fill_none(mask, False)
            n_events = ak.sum(region_masks[region_name])
            logging.info(f"Region {region_name}: {n_events} events")
        return region_masks

    def validate_regions(self, events: ak.Array, objects: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validation.get("check_orthogonality", True):
            return {"status": "skipped"}
        region_masks = self.apply_regions(events, objects)
        validation_results = {
            "status": "completed",
            "regions": {},
            "overlaps": {},
            "warnings": []
        }
        for region_name, mask in region_masks.items():
            n_events = ak.sum(mask)
            validation_results["regions"][region_name] = {
                "n_events": n_events,
                "fraction": float(n_events) / len(events) if len(events) > 0 else 0.0
            }
            min_events = self.settings.get("min_events", 10)
            if n_events < min_events:
                validation_results["warnings"].append(
                    f"Region {region_name} has only {n_events} events (minimum: {min_events})"
                )
        if self.validation.get("check_overlap", True):
            max_overlap = self.settings.get("max_overlap", 0.1)
            names = list(region_masks.keys())
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    m1 = region_masks[names[i]]
                    m2 = region_masks[names[j]]
                    overlap = ak.sum(m1 & m2)
                    total1 = ak.sum(m1)
                    total2 = ak.sum(m2)
                    if total1 > 0 and total2 > 0:
                        frac = float(overlap) / min(total1, total2)
                        validation_results["overlaps"][f"{names[i]}_{names[j]}"] = {
                            "n_overlap": overlap,
                            "fraction": frac,
                        }
                        if frac > max_overlap:
                            validation_results["warnings"].append(
                                f"High overlap between {names[i]} and {names[j]}: {frac:.2f}"
                            )
        return validation_results

    def get_region_summary(self) -> Dict[str, Any]:
        summary = {"n_regions": len(self.regions), "regions": {}}
        for region_name, region in self.regions.items():
            summary["regions"][region_name] = {
                "description": region.description,
                "n_cuts": len(region.cuts),
                "expected_backgrounds": region.expected_backgrounds,
                "blind_data": region.blind_data,
                "priority": region.priority,
                "transfer_factor": region.transfer_factor_to_SR,
            }
        return summary

    def get_signal_regions(self) -> List[str]:
        return [name for name in self.regions if ("SR" in name)]

    def get_control_regions(self) -> List[str]:
        return [name for name in self.regions if ("CR" in name)]

    def get_validation_regions(self) -> List[str]:
        return [name for name in self.regions if ("VR" in name)]
