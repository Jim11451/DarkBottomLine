"""
Histogram definitions and filling logic for DarkBottomLine framework.
"""

import awkward as ak
import numpy as np
from typing import Dict, Any, Optional, Union
import logging

try:
    import hist
    HIST_AVAILABLE = True
except ImportError:
    HIST_AVAILABLE = False
    logging.warning("hist library not available. Using fallback histogram implementation.")


class HistogramManager:
    """
    Manager class for creating and filling histograms.
    """

    def __init__(self):
        """Initialize histogram manager."""
        self.histograms = {}
        self.hist_specs = {}

    def define_histograms(self) -> Dict[str, Any]:
        """
        Define all histograms for the analysis.

        Returns:
            Dictionary of histogram specifications
        """
        if HIST_AVAILABLE:
            return self._define_hist_histograms()
        else:
            return self._define_fallback_histograms()

    def _define_hist_histograms(self) -> Dict[str, Any]:
        """Define histograms using hist library."""
        histograms = {}

        # MET histogram
        histograms["met"] = hist.Hist(
            hist.axis.Regular(50, 0, 500, name="met", label="MET [GeV]"),
            storage=hist.storage.Weight()
        )

        # Jet multiplicity
        histograms["n_jets"] = hist.Hist(
            hist.axis.Regular(10, 0, 10, name="n_jets", label="Number of Jets"),
            storage=hist.storage.Weight()
        )

        # B-jet multiplicity
        histograms["n_bjets"] = hist.Hist(
            hist.axis.Regular(6, 0, 6, name="n_bjets", label="Number of B-jets"),
            storage=hist.storage.Weight()
        )

        # Lepton multiplicity
        histograms["n_muons"] = hist.Hist(
            hist.axis.Regular(5, 0, 5, name="n_muons", label="Number of Muons"),
            storage=hist.storage.Weight()
        )

        histograms["n_electrons"] = hist.Hist(
            hist.axis.Regular(5, 0, 5, name="n_electrons", label="Number of Electrons"),
            storage=hist.storage.Weight()
        )

        histograms["n_taus"] = hist.Hist(
            hist.axis.Regular(5, 0, 5, name="n_taus", label="Number of Taus"),
            storage=hist.storage.Weight()
        )

        # Jet kinematics
        histograms["jet_pt"] = hist.Hist(
            hist.axis.Regular(50, 0, 500, name="jet_pt", label="Jet pT [GeV]"),
            storage=hist.storage.Weight()
        )

        histograms["jet_eta"] = hist.Hist(
            hist.axis.Regular(50, -3, 3, name="jet_eta", label="Jet η"),
            storage=hist.storage.Weight()
        )

        histograms["jet_phi"] = hist.Hist(
            hist.axis.Regular(50, -np.pi, np.pi, name="jet_phi", label="Jet φ"),
            storage=hist.storage.Weight()
        )

        # Lepton kinematics
        histograms["muon_pt"] = hist.Hist(
            hist.axis.Regular(50, 0, 200, name="muon_pt", label="Muon pT [GeV]"),
            storage=hist.storage.Weight()
        )

        histograms["muon_eta"] = hist.Hist(
            hist.axis.Regular(50, -2.5, 2.5, name="muon_eta", label="Muon η"),
            storage=hist.storage.Weight()
        )

        histograms["electron_pt"] = hist.Hist(
            hist.axis.Regular(50, 0, 200, name="electron_pt", label="Electron pT [GeV]"),
            storage=hist.storage.Weight()
        )

        histograms["electron_eta"] = hist.Hist(
            hist.axis.Regular(50, -2.5, 2.5, name="electron_eta", label="Electron η"),
            storage=hist.storage.Weight()
        )

        # Delta-R distributions
        histograms["dr_muon_jet"] = hist.Hist(
            hist.axis.Regular(50, 0, 5, name="dr_muon_jet", label="ΔR(μ, jet)"),
            storage=hist.storage.Weight()
        )

        histograms["dr_electron_jet"] = hist.Hist(
            hist.axis.Regular(50, 0, 5, name="dr_electron_jet", label="ΔR(e, jet)"),
            storage=hist.storage.Weight()
        )

        # B-tagging discriminants
        histograms["btag_deepjet"] = hist.Hist(
            hist.axis.Regular(50, 0, 1, name="btag_deepjet", label="DeepJet Score"),
            storage=hist.storage.Weight()
        )

        # DNN score histogram
        histograms["dnn_score"] = hist.Hist(
            hist.axis.Regular(50, 0, 1, name="dnn_score", label="DNN Score"),
            storage=hist.storage.Weight()
        )

        # Additional variables from StackPlotter analysis

        # MET phi
        histograms["met_phi"] = hist.Hist(
            hist.axis.Regular(50, -np.pi, np.pi, name="met_phi", label="MET φ"),
            storage=hist.storage.Weight()
        )

        # Recoil (for control regions)
        histograms["recoil"] = hist.Hist(
            hist.axis.Regular(50, 0, 500, name="recoil", label="Recoil [GeV]"),
            storage=hist.storage.Weight()
        )

        # Jet2 kinematics
        histograms["jet2_pt"] = hist.Hist(
            hist.axis.Regular(50, 0, 500, name="jet2_pt", label="Jet2 pT [GeV]"),
            storage=hist.storage.Weight()
        )

        histograms["jet2_eta"] = hist.Hist(
            hist.axis.Regular(50, -3, 3, name="jet2_eta", label="Jet2 η"),
            storage=hist.storage.Weight()
        )

        histograms["jet2_phi"] = hist.Hist(
            hist.axis.Regular(50, -np.pi, np.pi, name="jet2_phi", label="Jet2 φ"),
            storage=hist.storage.Weight()
        )

        histograms["jet2_deepcsv"] = hist.Hist(
            hist.axis.Regular(50, 0, 1, name="jet2_deepcsv", label="Jet2 DeepCSV"),
            storage=hist.storage.Weight()
        )

        # Jet3 kinematics
        histograms["jet3_pt"] = hist.Hist(
            hist.axis.Regular(50, 0, 500, name="jet3_pt", label="Jet3 pT [GeV]"),
            storage=hist.storage.Weight()
        )

        histograms["jet3_eta"] = hist.Hist(
            hist.axis.Regular(50, -3, 3, name="jet3_eta", label="Jet3 η"),
            storage=hist.storage.Weight()
        )

        histograms["jet3_phi"] = hist.Hist(
            hist.axis.Regular(50, -np.pi, np.pi, name="jet3_phi", label="Jet3 φ"),
            storage=hist.storage.Weight()
        )

        # Jet energy fractions and multiplicities
        histograms["jet1_nhad_ef"] = hist.Hist(
            hist.axis.Regular(50, 0, 1, name="jet1_nhad_ef", label="Jet1 NHadEF"),
            storage=hist.storage.Weight()
        )

        histograms["jet1_chad_ef"] = hist.Hist(
            hist.axis.Regular(50, 0, 1, name="jet1_chad_ef", label="Jet1 CHadEF"),
            storage=hist.storage.Weight()
        )

        histograms["jet1_cem_ef"] = hist.Hist(
            hist.axis.Regular(50, 0, 1, name="jet1_cem_ef", label="Jet1 CEmEF"),
            storage=hist.storage.Weight()
        )

        histograms["jet1_nem_ef"] = hist.Hist(
            hist.axis.Regular(50, 0, 1, name="jet1_nem_ef", label="Jet1 NEmEF"),
            storage=hist.storage.Weight()
        )

        histograms["jet1_cmulti"] = hist.Hist(
            hist.axis.Regular(20, 0, 20, name="jet1_cmulti", label="Jet1 CMulti"),
            storage=hist.storage.Weight()
        )

        histograms["jet1_nmultiplicity"] = hist.Hist(
            hist.axis.Regular(20, 0, 20, name="jet1_nmultiplicity", label="Jet1 NMultiplicity"),
            storage=hist.storage.Weight()
        )

        # Jet2 energy fractions
        histograms["jet2_nhad_ef"] = hist.Hist(
            hist.axis.Regular(50, 0, 1, name="jet2_nhad_ef", label="Jet2 NHadEF"),
            storage=hist.storage.Weight()
        )

        histograms["jet2_chad_ef"] = hist.Hist(
            hist.axis.Regular(50, 0, 1, name="jet2_chad_ef", label="Jet2 CHadEF"),
            storage=hist.storage.Weight()
        )

        histograms["jet2_cem_ef"] = hist.Hist(
            hist.axis.Regular(50, 0, 1, name="jet2_cem_ef", label="Jet2 CEmEF"),
            storage=hist.storage.Weight()
        )

        histograms["jet2_nem_ef"] = hist.Hist(
            hist.axis.Regular(50, 0, 1, name="jet2_nem_ef", label="Jet2 NEmEF"),
            storage=hist.storage.Weight()
        )

        histograms["jet2_cmulti"] = hist.Hist(
            hist.axis.Regular(20, 0, 20, name="jet2_cmulti", label="Jet2 CMulti"),
            storage=hist.storage.Weight()
        )

        histograms["jet2_nmultiplicity"] = hist.Hist(
            hist.axis.Regular(20, 0, 20, name="jet2_nmultiplicity", label="Jet2 NMultiplicity"),
            storage=hist.storage.Weight()
        )

        # Jet ratios and differences
        histograms["ratio_pt_jet21"] = hist.Hist(
            hist.axis.Regular(50, 0, 2, name="ratio_pt_jet21", label="Jet2/Jet1 pT Ratio"),
            storage=hist.storage.Weight()
        )

        histograms["dphi_jet12"] = hist.Hist(
            hist.axis.Regular(50, 0, np.pi, name="dphi_jet12", label="Δφ(Jet1,Jet2)"),
            storage=hist.storage.Weight()
        )

        histograms["deta_jet12"] = hist.Hist(
            hist.axis.Regular(50, 0, 6, name="deta_jet12", label="Δη(Jet1,Jet2)"),
            storage=hist.storage.Weight()
        )

        # Dijet masses
        histograms["m_jet1jet2"] = hist.Hist(
            hist.axis.Regular(50, 0, 1000, name="m_jet1jet2", label="M(Jet1,Jet2) [GeV]"),
            storage=hist.storage.Weight()
        )

        histograms["m_jet1jet3"] = hist.Hist(
            hist.axis.Regular(50, 0, 1000, name="m_jet1jet3", label="M(Jet1,Jet3) [GeV]"),
            storage=hist.storage.Weight()
        )

        # Jet eta matching
        histograms["isjet1_eta_match"] = hist.Hist(
            hist.axis.Regular(2, 0, 2, name="isjet1_eta_match", label="Jet1 η Match"),
            storage=hist.storage.Weight()
        )

        histograms["isjet2_eta_match"] = hist.Hist(
            hist.axis.Regular(2, 0, 2, name="isjet2_eta_match", label="Jet2 η Match"),
            storage=hist.storage.Weight()
        )

        # minΔφ and MET quality
        histograms["min_dphi"] = hist.Hist(
            hist.axis.Regular(50, 0, np.pi, name="min_dphi", label="min Δφ"),
            storage=hist.storage.Weight()
        )

        histograms["delta_pf_calo"] = hist.Hist(
            hist.axis.Regular(50, -100, 100, name="delta_pf_calo", label="pfMET - caloMET [GeV]"),
            storage=hist.storage.Weight()
        )

        # Primary vertices
        histograms["n_pv"] = hist.Hist(
            hist.axis.Regular(50, 0, 100, name="n_pv", label="Number of PV"),
            storage=hist.storage.Weight()
        )

        histograms["pu_npv"] = hist.Hist(
            hist.axis.Regular(50, 0, 100, name="pu_npv", label="PU NPV"),
            storage=hist.storage.Weight()
        )

        # Jet1 pT / MET ratio
        histograms["rjet1_pt_met"] = hist.Hist(
            hist.axis.Regular(50, 0, 5, name="rjet1_pt_met", label="Jet1 pT / MET"),
            storage=hist.storage.Weight()
        )

        # Production category and CTS value
        histograms["prod_cat"] = hist.Hist(
            hist.axis.Regular(10, 0, 10, name="prod_cat", label="Production Category"),
            storage=hist.storage.Weight()
        )

        histograms["cts_value"] = hist.Hist(
            hist.axis.Regular(50, -1, 1, name="cts_value", label="CTS Value"),
            storage=hist.storage.Weight()
        )

        # Lepton kinematics for control regions
        histograms["lep1_pt"] = hist.Hist(
            hist.axis.Regular(50, 0, 200, name="lep1_pt", label="Leading Lepton pT [GeV]"),
            storage=hist.storage.Weight()
        )

        histograms["lep1_phi"] = hist.Hist(
            hist.axis.Regular(50, -np.pi, np.pi, name="lep1_phi", label="Leading Lepton φ"),
            storage=hist.storage.Weight()
        )

        histograms["lep2_pt"] = hist.Hist(
            hist.axis.Regular(50, 0, 200, name="lep2_pt", label="Subleading Lepton pT [GeV]"),
            storage=hist.storage.Weight()
        )

        histograms["lep2_phi"] = hist.Hist(
            hist.axis.Regular(50, -np.pi, np.pi, name="lep2_phi", label="Subleading Lepton φ"),
            storage=hist.storage.Weight()
        )

        histograms["dphi_lep1_met"] = hist.Hist(
            hist.axis.Regular(50, 0, np.pi, name="dphi_lep1_met", label="Δφ(Lepton1,MET)"),
            storage=hist.storage.Weight()
        )

        # W/Z masses and pT
        histograms["w_mass"] = hist.Hist(
            hist.axis.Regular(50, 0, 200, name="w_mass", label="W Mass [GeV]"),
            storage=hist.storage.Weight()
        )

        histograms["w_pt"] = hist.Hist(
            hist.axis.Regular(50, 0, 500, name="w_pt", label="W pT [GeV]"),
            storage=hist.storage.Weight()
        )

        histograms["z_mass"] = hist.Hist(
            hist.axis.Regular(50, 0, 200, name="z_mass", label="Z Mass [GeV]"),
            storage=hist.storage.Weight()
        )

        histograms["z_pt"] = hist.Hist(
            hist.axis.Regular(50, 0, 500, name="z_pt", label="Z pT [GeV]"),
            storage=hist.storage.Weight()
        )

        # Transverse mass and invariant mass
        histograms["mt"] = hist.Hist(
            hist.axis.Regular(50, 0, 200, name="mt", label="Transverse Mass [GeV]"),
            storage=hist.storage.Weight()
        )

        histograms["mll"] = hist.Hist(
            hist.axis.Regular(50, 0, 200, name="mll", label="Dilepton Mass [GeV]"),
            storage=hist.storage.Weight()
        )

        # MET quality
        histograms["met_quality"] = hist.Hist(
            hist.axis.Regular(50, -2, 2, name="met_quality", label="MET Quality"),
            storage=hist.storage.Weight()
        )

        return histograms

    def _define_fallback_histograms(self) -> Dict[str, Any]:
        """Define histograms using fallback implementation."""
        histograms = {}

        # Simple histogram specifications for fallback
        histograms["met"] = {
            "bins": np.linspace(0, 500, 51),
            "label": "MET [GeV]",
            "values": [],
            "weights": []
        }

        histograms["n_jets"] = {
            "bins": np.arange(0, 11),
            "label": "Number of Jets",
            "values": [],
            "weights": []
        }

        histograms["n_bjets"] = {
            "bins": np.arange(0, 7),
            "label": "Number of B-jets",
            "values": [],
            "weights": []
        }

        return histograms

    def fill_histograms(
        self,
        events: ak.Array,
        objects: Dict[str, Any],
        weights: Union[ak.Array, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Fill all histograms with event data.

        Args:
            events: Awkward Array of events
            objects: Dictionary containing selected objects
            weights: Event weights

        Returns:
            Dictionary of filled histograms
        """
        if HIST_AVAILABLE:
            return self._fill_hist_histograms(events, objects, weights)
        else:
            return self._fill_fallback_histograms(events, objects, weights)

    def _fill_hist_histograms(
        self,
        events: ak.Array,
        objects: Dict[str, Any],
        weights: Union[ak.Array, np.ndarray]
    ) -> Dict[str, Any]:
        """Fill histograms using hist library."""
        histograms = self.define_histograms()

        # Handle empty events
        if len(events) == 0:
            return histograms

        # Convert weights to numpy array
        if isinstance(weights, ak.Array):
            weights = ak.to_numpy(weights)

        # Fill MET histogram
        histograms["met"].fill(met=events["PFMET_pt"], weight=weights)

        # Fill multiplicity histograms
        n_jets = ak.num(objects["jets"], axis=1)
        n_bjets = ak.num(objects["bjets"], axis=1)
        n_muons = ak.num(objects["muons"], axis=1)
        n_electrons = ak.num(objects["electrons"], axis=1)
        n_taus = ak.num(objects["taus"], axis=1)

        histograms["n_jets"].fill(n_jets=n_jets, weight=weights)
        histograms["n_bjets"].fill(n_bjets=n_bjets, weight=weights)
        histograms["n_muons"].fill(n_muons=n_muons, weight=weights)
        histograms["n_electrons"].fill(n_electrons=n_electrons, weight=weights)
        histograms["n_taus"].fill(n_taus=n_taus, weight=weights)

        # Fill jet kinematics
        if len(ak.flatten(objects["jets"])) > 0:
            jet_pt = ak.flatten(objects["jets"].pt)
            jet_eta = ak.flatten(objects["jets"].eta)
            jet_phi = ak.flatten(objects["jets"].phi)

            # Repeat weights for each jet
            jet_weights = ak.flatten(ak.broadcast_arrays(weights, objects["jets"].pt)[0])

            histograms["jet_pt"].fill(jet_pt=jet_pt, weight=jet_weights)
            histograms["jet_eta"].fill(jet_eta=jet_eta, weight=jet_weights)
            histograms["jet_phi"].fill(jet_phi=jet_phi, weight=jet_weights)

            # Fill b-tagging discriminant
            if hasattr(objects["jets"], "btagDeepFlavB"):
                btag_score = ak.flatten(objects["jets"].btagDeepFlavB)
                histograms["btag_deepjet"].fill(btag_deepjet=btag_score, weight=jet_weights)

        # Fill muon kinematics
        if len(ak.flatten(objects["muons"])) > 0:
            muon_pt = ak.flatten(objects["muons"].pt)
            muon_eta = ak.flatten(objects["muons"].eta)

            muon_weights = ak.flatten(ak.broadcast_arrays(weights, objects["muons"].pt)[0])

            histograms["muon_pt"].fill(muon_pt=muon_pt, weight=muon_weights)
            histograms["muon_eta"].fill(muon_eta=muon_eta, weight=muon_weights)

        # Fill electron kinematics
        if len(ak.flatten(objects["electrons"])) > 0:
            electron_pt = ak.flatten(objects["electrons"].pt)
            electron_eta = ak.flatten(objects["electrons"].eta)

            electron_weights = ak.flatten(ak.broadcast_arrays(weights, objects["electrons"].pt)[0])

            histograms["electron_pt"].fill(electron_pt=electron_pt, weight=electron_weights)
            histograms["electron_eta"].fill(electron_eta=electron_eta, weight=electron_weights)

        # Fill Delta-R distributions
        if len(ak.flatten(objects["muons"])) > 0 and len(ak.flatten(objects["jets"])) > 0:
            # Calculate Delta-R between muons and jets
            # This is a simplified version - would need proper 2D Delta-R calculation
            pass  # TODO: Implement Delta-R calculation

        return histograms

    def _fill_fallback_histograms(
        self,
        events: ak.Array,
        objects: Dict[str, Any],
        weights: Union[ak.Array, np.ndarray]
    ) -> Dict[str, Any]:
        """Fill histograms using fallback implementation."""
        histograms = self.define_histograms()

        # Handle empty events
        if len(events) == 0:
            return histograms

        # Convert weights to numpy array
        if isinstance(weights, ak.Array):
            weights = ak.to_numpy(weights)

        # Fill MET histogram
        met_values = ak.to_numpy(events["PFMET_pt"])
        histograms["met"]["values"].extend(met_values)
        histograms["met"]["weights"].extend(weights)

        # Fill multiplicity histograms
        n_jets = ak.to_numpy(ak.num(objects["jets"], axis=1))
        n_bjets = ak.to_numpy(ak.num(objects["bjets"], axis=1))

        histograms["n_jets"]["values"].extend(n_jets)
        histograms["n_jets"]["weights"].extend(weights)

        histograms["n_bjets"]["values"].extend(n_bjets)
        histograms["n_bjets"]["weights"].extend(weights)

        return histograms

    def get_histogram_statistics(self, histograms: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all histograms.

        Args:
            histograms: Dictionary of histograms

        Returns:
            Dictionary with statistics for each histogram
        """
        stats = {}

        for name, hist in histograms.items():
            if HIST_AVAILABLE and hasattr(hist, 'values'):
                # Hist library histogram
                stats[name] = {
                    "entries": int(hist.values().sum()),
                    "mean": float(hist.values().mean()),
                    "std": float(hist.values().std()),
                }
            else:
                # Fallback histogram
                if "values" in hist and len(hist["values"]) > 0:
                    values = np.array(hist["values"])
                    weights = np.array(hist["weights"])

                    # Handle empty arrays
                    if len(values) == 0:
                        stats[name] = {
                            "entries": 0,
                            "mean": 0.0,
                            "std": 0.0,
                        }
                    else:
                        stats[name] = {
                            "entries": int(np.sum(weights)),
                            "mean": float(np.average(values, weights=weights)),
                            "std": float(np.sqrt(np.average((values - np.average(values, weights=weights))**2, weights=weights))),
                        }
                else:
                    stats[name] = {
                        "entries": 0,
                        "mean": 0.0,
                        "std": 0.0,
                    }

        return stats

    def save_histograms(self, histograms: Dict[str, Any], output_file: str):
        """
        Save histograms to file.

        Args:
            histograms: Dictionary of histograms
            output_file: Output file path
        """
        if HIST_AVAILABLE:
            # Save using hist library
            import pickle
            with open(output_file, 'wb') as f:
                pickle.dump(histograms, f)
        else:
            # Save fallback histograms
            import json
            # Convert to JSON-serializable format
            json_histograms = {}
            for name, hist in histograms.items():
                json_histograms[name] = {
                    "bins": hist["bins"].tolist(),
                    "label": hist["label"],
                    "values": hist["values"],
                    "weights": hist["weights"]
                }
            with open(output_file, 'w') as f:
                json.dump(json_histograms, f)
