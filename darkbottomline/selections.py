"""
Event selection functions for DarkBottomLine framework.
"""

import awkward as ak
import numpy as np
from typing import Dict, Any, List, Tuple


def pass_triggers(events: ak.Array, trigger_paths: List[str]) -> ak.Array:
    """
    Check if events pass any of the specified trigger paths.

    Args:
        events: Awkward Array of events
        trigger_paths: List of trigger path names to check

    Returns:
        Boolean mask for events passing triggers
    """
    if not trigger_paths:
        return ak.ones_like(events["event"], dtype=bool)

    # Initialize with False
    trigger_mask = ak.zeros_like(events["event"], dtype=bool)

    # Check each trigger path
    for trigger in trigger_paths:
        if trigger in events.fields:
            trigger_branch = events[trigger]
            trigger_mask = trigger_mask | trigger_branch

    return trigger_mask


def pass_met_filters(events: ak.Array, filter_names: List[str]) -> ak.Array:
    """
    Check if events pass all specified MET filters.

    Args:
        events: Awkward Array of events
        filter_names: List of MET filter names to check

    Returns:
        Boolean mask for events passing all filters
    """
    if not filter_names:
        return ak.ones_like(events["event"], dtype=bool)

    # Initialize with True
    filter_mask = ak.ones_like(events["event"], dtype=bool)

    # Check each filter
    for filter_name in filter_names:
        if filter_name in events.fields:
            filter_branch = events[filter_name]
            filter_mask = filter_mask & filter_branch

    return filter_mask


def select_events(
    events: ak.Array,
    objects: Dict[str, Any],
    config: Dict[str, Any]
) -> ak.Array:
    """
    Apply event-level selection cuts.

    Args:
        events: Awkward Array of events
        objects: Dictionary containing selected objects
        config: Configuration dictionary with event selection cuts

    Returns:
        Boolean mask for events passing selection
    """
    selection = config["event_selection"]

    # Count objects per event
    n_muons = ak.num(objects["muons"], axis=1)
    n_electrons = ak.num(objects["electrons"], axis=1)
    n_taus = ak.num(objects["taus"], axis=1)
    n_jets = ak.num(objects["jets"], axis=1)
    n_bjets = ak.num(objects["bjets"], axis=1)

    # Apply multiplicity cuts
    muon_cut = (n_muons >= selection["min_muons"]) & (n_muons <= selection["max_muons"])
    electron_cut = (n_electrons >= selection["min_electrons"]) & (n_electrons <= selection["max_electrons"])
    tau_cut = (n_taus >= selection["min_taus"]) & (n_taus <= selection["max_taus"])
    jet_cut = n_jets >= selection["min_jets"]
    bjet_cut = n_bjets >= selection["min_bjets"]

    # MET or Recoil cut: pass if either MET > met_min or Recoil > met_min
    met_pt = events["PFMET_pt"] if "PFMET_pt" in events.fields else events["MET_pt"]
    met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]
    met_min = selection["met_min"]
    met_cut = met_pt > met_min

    # Recoil = | -( pTmiss + sum pT(leptons) ) | (same convention as region Recoil)
    muons = objects.get("muons", ak.Array([]))
    electrons = objects.get("electrons", ak.Array([]))
    lep_pt_sum = ak.zeros_like(met_pt)
    try:
        if len(ak.flatten(muons)) > 0:
            lep_pt_sum = lep_pt_sum + ak.sum(muons.pt, axis=1)
        if len(ak.flatten(electrons)) > 0:
            lep_pt_sum = lep_pt_sum + ak.sum(electrons.pt, axis=1)
    except (Exception, BaseException):
        pass
    recoil_px = -(met_pt * np.cos(met_phi) + ak.fill_none(lep_pt_sum, 0.0))
    recoil_py = -(met_pt * np.sin(met_phi))
    recoil = np.sqrt(recoil_px**2 + recoil_py**2)
    recoil_cut = recoil > met_min

    met_or_recoil_cut = met_cut | recoil_cut

    # Leading jet pT cut (paper: leading jet pT > 100 GeV)
    jet1_pt_min = selection.get("jet1_pt_min")
    if jet1_pt_min is not None:
        jets = objects.get("jets", ak.Array([]))
        if len(ak.flatten(jets)) > 0:
            leading_jet_pt = ak.fill_none(ak.max(jets.pt, axis=1), 0.0)
            jet1_pt_cut = leading_jet_pt > jet1_pt_min
        else:
            jet1_pt_cut = ak.zeros_like(events["event"], dtype=bool)
    else:
        jet1_pt_cut = ak.ones_like(events["event"], dtype=bool)

    # min DeltaPhi(jet, pTmiss) > threshold (paper: > 0.5 for QCD multijet suppression)
    delta_phi_min = selection.get("delta_phi_min")
    if delta_phi_min is not None:
        jets = objects.get("jets", ak.Array([]))
        met_phi = events["PFMET_phi"] if "PFMET_phi" in events.fields else events["MET_phi"]
        if len(ak.flatten(jets)) > 0:
            dphi = np.abs(jets.phi - met_phi)
            dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)
            min_dphi = ak.fill_none(ak.min(dphi, axis=1), 0.0)
            delta_phi_cut = min_dphi > delta_phi_min
        else:
            delta_phi_cut = ak.zeros_like(events["event"], dtype=bool)
    else:
        delta_phi_cut = ak.ones_like(events["event"], dtype=bool)

    # Combine all cuts (met_min: either MET or Recoil above threshold)
    event_mask = (
        muon_cut & electron_cut & tau_cut & jet_cut & bjet_cut & met_or_recoil_cut
        & jet1_pt_cut & delta_phi_cut
    )

    return event_mask


def get_cutflow(
    events: ak.Array,
    objects: Dict[str, Any],
    config: Dict[str, Any]
) -> Dict[str, int]:
    """
    Calculate cutflow for event selection.

    Args:
        events: Awkward Array of events
        objects: Dictionary containing selected objects
        config: Configuration dictionary

    Returns:
        Dictionary with cut names and event counts
    """
    n_total = len(events)

    # Trigger selection
    trigger_mask = pass_triggers(events, config["triggers"]["MET"])
    n_trigger = ak.sum(trigger_mask)

    # MET filter selection
    filter_mask = pass_met_filters(events, config["met_filters"])
    n_filters = ak.sum(trigger_mask & filter_mask)

    # Object selection
    n_muons = ak.sum(ak.num(objects["muons"], axis=1) > 0)
    n_electrons = ak.sum(ak.num(objects["electrons"], axis=1) > 0)
    n_taus = ak.sum(ak.num(objects["taus"], axis=1) > 0)
    n_jets = ak.sum(ak.num(objects["jets"], axis=1) > 0)
    n_bjets = ak.sum(ak.num(objects["bjets"], axis=1) > 0)

    # Event selection
    event_mask = select_events(events, objects, config)
    n_selected = ak.sum(event_mask)

    return {
        "Total events": n_total,
        "Pass trigger": n_trigger,
        "Pass filters": n_filters,
        "Has muons": n_muons,
        "Has electrons": n_electrons,
        "Has taus": n_taus,
        "Has jets": n_jets,
        "Has bjets": n_bjets,
        "Final selection": n_selected,
    }


def apply_selection(
    events: ak.Array,
    objects: Dict[str, Any],
    config: Dict[str, Any]
) -> Tuple[ak.Array, Dict[str, Any], Dict[str, int]]:
    """
    Apply complete event selection including triggers, filters, and cuts.

    Args:
        events: Awkward Array of events
        objects: Dictionary containing selected objects
        config: Configuration dictionary

    Returns:
        Tuple of (selected_events, selected_objects, cutflow)
    """
    combined_trigger_mask = ak.zeros_like(events["event"], dtype=bool)
    for trigger_type, trigger_paths in config["triggers"].items():
        if trigger_paths: # Only apply if there are paths for this type
            type_mask = pass_triggers(events, trigger_paths)
            combined_trigger_mask = combined_trigger_mask | type_mask
    trigger_mask = combined_trigger_mask

    print("  Applying MET filter selection...")
    # Apply MET filter selection
    filter_mask = pass_met_filters(events, config["met_filters"])
    print(f"    Events passing MET filters: {ak.sum(filter_mask)}")

    print("  Applying event selection...")
    # Apply event selection
    event_mask = select_events(events, objects, config)
    print(f"    Events passing event selection: {ak.sum(event_mask)}")

    # Combine all masks
    final_mask = trigger_mask & filter_mask & event_mask
    print(f"    Events passing all selections: {ak.sum(final_mask)}")

    # Apply selection to events and objects
    selected_events = events[final_mask]
    selected_objects = {}
    for key, obj in objects.items():
        if isinstance(obj, ak.Array):
            selected_objects[key] = obj[final_mask]
        else:
            selected_objects[key] = obj

    # Calculate cutflow
    cutflow = get_cutflow(events, objects, config)

    return selected_events, selected_objects, cutflow
