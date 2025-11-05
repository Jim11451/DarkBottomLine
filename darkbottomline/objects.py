"""
Physics object selection and cleaning functions for DarkBottomLine framework.
"""

import awkward as ak
import numpy as np
from typing import Dict, Any, Tuple


def select_muons(events: ak.Array, config: Dict[str, Any]) -> ak.Array:
    """
    Select muons based on configuration cuts.

    Args:
        events: Awkward Array of events
        config: Configuration dictionary with muon selection cuts

    Returns:
        Boolean mask for selected muons
    """
    # Build muon collection from flat branches
    muons = ak.zip({
        "pt": events["Muon_pt"],
        "eta": events["Muon_eta"],
        "phi": events["Muon_phi"],
        "tightId": events["Muon_tightId"],
        "pfIsoId": events["Muon_pfIsoId"],
    })

    # Basic kinematic cuts
    pt_mask = muons.pt > config["pt_min"]
    eta_mask = abs(muons.eta) < config["eta_max"]

    # ID and isolation cuts (simplified - would need actual CMS ID implementation)
    id_mask = muons.tightId == 1  # tightId branch exists
    iso_mask = muons.pfIsoId >= 3  # pfIsoId branch exists (3 = tight)

    # Combine all cuts
    selection_mask = pt_mask & eta_mask & id_mask & iso_mask

    return selection_mask


def select_electrons(events: ak.Array, config: Dict[str, Any]) -> ak.Array:
    """
    Select electrons based on configuration cuts.

    Args:
        events: Awkward Array of events
        config: Configuration dictionary with electron selection cuts

    Returns:
        Boolean mask for selected electrons
    """
    # Build electron collection from flat branches
    electrons = ak.zip({
        "pt": events["Electron_pt"],
        "eta": events["Electron_eta"],
        "phi": events["Electron_phi"],
        "cutBased": events["Electron_cutBased"],
        "mvaIso_WP80": events["Electron_mvaIso_WP80"],
    })

    # Basic kinematic cuts
    pt_mask = electrons.pt > config["pt_min"]
    eta_mask = abs(electrons.eta) < config["eta_max"]

    # ID and isolation cuts (simplified - would need actual CMS ID implementation)
    id_mask = electrons.cutBased >= 4  # cutBased branch exists (4 = tight)
    iso_mask = electrons.mvaIso_WP80 == 1  # mvaIso_WP80 branch exists (1 = pass)

    # Combine all cuts
    selection_mask = pt_mask & eta_mask & id_mask & iso_mask

    return selection_mask


def select_taus(events: ak.Array, config: Dict[str, Any]) -> ak.Array:
    """
    Select taus based on configuration cuts.

    Args:
        events: Awkward Array of events
        config: Configuration dictionary with tau selection cuts

    Returns:
        Boolean mask for selected taus
    """
    # Build tau collection from flat branches
    taus = ak.zip({
        "pt": events["Tau_pt"],
        "eta": events["Tau_eta"],
        "phi": events["Tau_phi"],
        "idDeepTau2018v2p5VSjet": events["Tau_idDeepTau2018v2p5VSjet"],
        "decayMode": events["Tau_decayMode"],
    })

    # Basic kinematic cuts
    pt_mask = taus.pt > config["pt_min"]
    eta_mask = abs(taus.eta) < config["eta_max"]

    # ID and decay mode cuts (simplified - would need actual CMS ID implementation)
    id_mask = taus.idDeepTau2018v2p5VSjet >= 16  # idDeepTau2018v2p5VSjet branch exists (16 = tight)
    # Check if decay mode is in allowed modes
    decay_mode_mask = ak.zeros_like(taus.pt, dtype=bool)
    for mode in config["decay_modes"]:
        decay_mode_mask = decay_mode_mask | (taus.decayMode == mode)

    # Combine all cuts
    selection_mask = pt_mask & eta_mask & id_mask & decay_mode_mask

    return selection_mask


def select_jets(events: ak.Array, config: Dict[str, Any]) -> ak.Array:
    """
    Select AK4 jets based on configuration cuts.

    Args:
        events: Awkward Array of events
        config: Configuration dictionary with jet selection cuts

    Returns:
        Boolean mask for selected jets
    """
    # Build jet collection from flat branches
    jets = ak.zip({
        "pt": events["Jet_pt"],
        "eta": events["Jet_eta"],
        "phi": events["Jet_phi"],
        "puIdDisc": events["Jet_puIdDisc"],
        "btagDeepFlavB": events["Jet_btagDeepFlavB"],
    })

    # Basic kinematic cuts
    pt_mask = jets.pt > config["pt_min"]
    eta_mask = abs(jets.eta) < config["eta_max"]

    # Jet ID cut (simplified - would need actual CMS jet ID implementation)
    id_mask = jets.puIdDisc >= 0  # puIdDisc branch exists (>= 0 = pass)

    # Combine all cuts
    selection_mask = pt_mask & eta_mask & id_mask

    return selection_mask


def select_fatjets(events: ak.Array, config: Dict[str, Any]) -> ak.Array:
    """
    Select AK8 fat jets based on configuration cuts.

    Args:
        events: Awkward Array of events
        config: Configuration dictionary with fat jet selection cuts

    Returns:
        Boolean mask for selected fat jets
    """
    # Build fat jet collection from flat branches
    fatjets = ak.zip({
        "pt": events["FatJet_pt"],
        "eta": events["FatJet_eta"],
        "phi": events["FatJet_phi"],
        "mass": events["FatJet_mass"],
    })

    # Basic kinematic cuts
    pt_mask = fatjets.pt > config["pt_min"]
    eta_mask = abs(fatjets.eta) < config["eta_max"]

    # Fat jet ID cut (simplified - would need actual CMS fat jet ID implementation)
    # For now, just use basic kinematic cuts
    id_mask = ak.ones_like(fatjets.pt, dtype=bool)

    # Combine all cuts
    selection_mask = pt_mask & eta_mask & id_mask

    return selection_mask


def clean_jets_from_leptons(
    jets: ak.Array,
    leptons: ak.Array,
    dr_min: float = 0.4
) -> ak.Array:
    """
    Remove jets that are too close to selected leptons.

    Args:
        jets: Selected jets
        leptons: Selected leptons (muons, electrons, or taus)
        dr_min: Minimum Delta-R separation

    Returns:
        Boolean mask for jets that pass cleaning
    """
    if len(ak.flatten(leptons)) == 0:
        # No leptons, all jets pass
        return ak.ones_like(jets.pt, dtype=bool)

    # Calculate Delta-R between jets and all leptons
    # This is a simplified version - would need proper 2D Delta-R calculation
    jet_eta = jets.eta
    jet_phi = jets.phi
    lep_eta = ak.flatten(leptons.eta)
    lep_phi = ak.flatten(leptons.phi)

    # For each jet, check if it's far enough from any lepton
    # This is a placeholder - actual implementation would use proper Delta-R
    dr_mask = ak.ones_like(jets.pt, dtype=bool)

    # TODO: Implement proper Delta-R calculation
    # For now, return all jets as passing
    return dr_mask


def get_bjet_mask(jets: ak.Array, config: Dict[str, Any]) -> ak.Array:
    """
    Get b-tagging mask for jets based on working point.

    Args:
        jets: Selected jets
        config: Configuration dictionary with b-tagging parameters

    Returns:
        Boolean mask for b-tagged jets
    """
    algorithm = config["algorithm"]
    wp = config["wp"]

    if algorithm == "deepJet":
        if wp == "loose":
            return jets.btagDeepFlavB > 0.0490
        elif wp == "medium":
            return jets.btagDeepFlavB > 0.2783
        elif wp == "tight":
            return jets.btagDeepFlavB > 0.7100
    elif algorithm == "deepCSV":
        if wp == "loose":
            return jets.btagDeepB > 0.1208
        elif wp == "medium":
            return jets.btagDeepB > 0.4941
        elif wp == "tight":
            return jets.btagDeepB > 0.8001

    # Default to medium working point
    return jets.btagDeepFlavB > 0.2783


def build_objects(events: ak.Array, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build all physics objects with selection and cleaning applied.

    Args:
        events: Awkward Array of events
        config: Full configuration dictionary

    Returns:
        Dictionary containing selected objects and masks
    """
    print("  Building object collections from flat branches...")

    # Select objects
    print("  Selecting muons...")
    muon_mask = select_muons(events, config["objects"]["muons"])
    print(f"    Muons selected: {ak.sum(muon_mask)}")

    print("  Selecting electrons...")
    electron_mask = select_electrons(events, config["objects"]["electrons"])
    print(f"    Electrons selected: {ak.sum(electron_mask)}")

    print("  Selecting taus...")
    tau_mask = select_taus(events, config["objects"]["taus"])
    print(f"    Taus selected: {ak.sum(tau_mask)}")

    print("  Selecting jets...")
    jet_mask = select_jets(events, config["objects"]["jets"])
    print(f"    Jets selected: {ak.sum(jet_mask)}")

    print("  Selecting fat jets...")
    fatjet_mask = select_fatjets(events, config["objects"]["fatjets"])
    print(f"    Fat jets selected: {ak.sum(fatjet_mask)}")

    # Build collections from flat branches
    muons = ak.zip({
        "pt": events["Muon_pt"],
        "eta": events["Muon_eta"],
        "phi": events["Muon_phi"],
        "tightId": events["Muon_tightId"],
        "pfIsoId": events["Muon_pfIsoId"],
    })

    electrons = ak.zip({
        "pt": events["Electron_pt"],
        "eta": events["Electron_eta"],
        "phi": events["Electron_phi"],
        "cutBased": events["Electron_cutBased"],
        "mvaIso_WP80": events["Electron_mvaIso_WP80"],
    })

    taus = ak.zip({
        "pt": events["Tau_pt"],
        "eta": events["Tau_eta"],
        "phi": events["Tau_phi"],
        "idDeepTau2018v2p5VSjet": events["Tau_idDeepTau2018v2p5VSjet"],
        "decayMode": events["Tau_decayMode"],
    })

    jets = ak.zip({
        "pt": events["Jet_pt"],
        "eta": events["Jet_eta"],
        "phi": events["Jet_phi"],
        "puIdDisc": events["Jet_puIdDisc"],
        "btagDeepFlavB": events["Jet_btagDeepFlavB"],
    })

    fatjets = ak.zip({
        "pt": events["FatJet_pt"],
        "eta": events["FatJet_eta"],
        "phi": events["FatJet_phi"],
        "mass": events["FatJet_mass"],
    })

    # Apply masks to get selected objects
    selected_muons = muons[muon_mask]
    selected_electrons = electrons[electron_mask]
    selected_taus = taus[tau_mask]
    selected_jets = jets[jet_mask]
    selected_fatjets = fatjets[fatjet_mask]

    # Clean jets from leptons
    print("  Cleaning jets from leptons...")
    all_leptons = ak.concatenate([
        selected_muons, selected_electrons, selected_taus
    ], axis=1)

    jet_cleaning_mask = clean_jets_from_leptons(
        selected_jets,
        all_leptons,
        config["cleaning"]["dr_muon_jet"]
    )

    # Apply jet cleaning
    cleaned_jets = selected_jets[jet_cleaning_mask]
    print(f"    Jets after cleaning: {ak.sum(ak.num(cleaned_jets, axis=1))}")

    # Get b-tagging mask for cleaned jets
    print("  Applying b-tagging...")
    bjet_mask = get_bjet_mask(cleaned_jets, config["btagging"])
    print(f"    B-jets identified: {ak.sum(ak.num(cleaned_jets[bjet_mask], axis=1))}")

    print("  Object building complete!")
    return {
        "muons": selected_muons,
        "electrons": selected_electrons,
        "taus": selected_taus,
        "jets": cleaned_jets,
        "fatjets": selected_fatjets,
        "bjets": cleaned_jets[bjet_mask],
        "muon_mask": muon_mask,
        "electron_mask": electron_mask,
        "tau_mask": tau_mask,
        "jet_mask": jet_mask,
        "fatjet_mask": fatjet_mask,
        "bjet_mask": bjet_mask,
    }
