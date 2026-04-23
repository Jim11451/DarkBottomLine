#!/usr/bin/env python3
"""
Standalone stacked plotting tool for event-level output PKL/ROOT files.

Features:
- Reads event-level PKL format (keys: n_events/events/objects)
- Reads ROOT files (Events tree with branches, Metadata tree with n_events)
- Merges multiple PKL/ROOT files inside each input folder
- Supports multiple background folders and optional data folder
- Draws log-scale stacked background plot with data overlay
- Normalizes all histograms by n_events and luminosity
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import uproot


@dataclass
class SubSampleData:
    process: str
    n_events: int
    objects: Dict[str, Any]
    cross_section: Optional[float] = None  # pb


@dataclass
class SampleData:
    name: str
    n_events: int
    objects: Dict[str, Any]
    luminosity: float = 1.0  # Luminosity normalization factor
    cross_section: Optional[float] = None  # Cross-section in pb; None means no cross-section weighting
    metadata_n_events_available: bool = True  # Whether n_events is from metadata (for ROOT inputs)
    sub_samples: Optional[List[SubSampleData]] = None  # Optional per-process components for mixed-pt samples


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _load_xsection_json(json_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Load cross-section data from JSON file.
    
    Expected format:
    {
        "ProcessName": [
            {"year": "2022", "process": "...", "xsection": 123.45, ...},
            ...
        ],
        ...
    }
    """
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Could not load cross-section JSON {json_path}: {e}")
        return {}


def _extract_xsection_by_year(
    xsection_data: Dict[str, List[Dict[str, Any]]], 
    sample_name: str,
    year: str
) -> Optional[float]:
    """Extract cross-section (in pb) for a given sample and year.
    
    Searches for matching process in xsection_data and returns cross-section in pb.
    Returns None if not found.
    """
    if not xsection_data:
        return None
    
    # Try to find the process category from sample name
    # Sample names are usually like: newDIBOSON_2022_EVENTSELECTION or newWtoLNU-2Jets_2022_EVENTSELECTION
    # Remove prefixes and suffixes
    sample_base = sample_name
    if sample_base.startswith("new"):
        sample_base = sample_base[3:]
    # Remove year suffix and _EVENTSELECTION
    import re as regex
    sample_base = regex.sub(r"_20\d{2}(EE)?.*$", "", sample_base)
    # Normalize case for matching (WtoLNu vs WtoLNU)
    sample_base_lower = sample_base.lower()
    
    alias_map = {
        "wtolnu-2jets": "WtoLNu-2Jets",
        "zto2nu-2jets": "Zto2Nu-2Jets",
        "dyto2l-2jets": "DYto2L-2Jets",
        "top": "Top",
        "diboson": "DIBOSON",
        "singletop": "SingleTop",
        "smhiggs": "SMHiggs",
    }
    mapped_category = alias_map.get(sample_base_lower, sample_base)

    # Search in xsection_data for matching process
    for process_category, entries in xsection_data.items():
        # Try case-insensitive match for process category
        if mapped_category.lower() == process_category.lower():
            for entry in entries:
                if entry.get("year") == year:
                    xsec = entry.get("xsection")
                    if xsec is not None:
                        return float(xsec)  # Return in pb
    return None


def _build_xsection_process_lookup(
    xsection_data: Dict[str, List[Dict[str, Any]]],
    year: str,
) -> Dict[str, float]:
    """Build a mapping from full_dataset process name to cross-section in pb for a given year."""
    process_to_xsec: Dict[str, float] = {}
    for entries in xsection_data.values():
        for entry in entries:
            if entry.get("year") != year:
                continue
            full_dataset = entry.get("full_dataset")
            xsec = entry.get("xsection")
            if not full_dataset or xsec is None:
                continue
            process_to_xsec[str(full_dataset)] = float(xsec)
    return process_to_xsec


def _normalize_xsection_name(name: str) -> str:
    value = Path(name).stem
    if value.startswith("new"):
        value = value[3:]
    value = re.sub(r"_20\d{2}(EE)?_EVENTSELECTION", "", value)
    value = re.sub(r"_20\d{2}(EE)?$", "", value)
    # DY JSON keys can include MLL tags while merged ROOT names may omit them.
    value = re.sub(r"_MLL-\d+", "", value, flags=re.IGNORECASE)
    value = value.replace("_EVENTSELECTION", "")
    value = value.replace("singletop", "singleTop")
    value = value.replace("smhiggs", "smHiggs")
    return value.lower()


def _match_root_file_to_process(root_file_name: str, process_to_xsec: Dict[str, float]) -> Optional[Tuple[str, float]]:
    """Match ROOT file name to JSON full_dataset key using normalized longest-prefix match."""
    stem = _normalize_xsection_name(root_file_name)
    best_key: Optional[str] = None
    for key in process_to_xsec.keys():
        key_norm = _normalize_xsection_name(key)
        if stem.startswith(key_norm) or key_norm.startswith(stem):
            if best_key is None or len(key_norm) > len(_normalize_xsection_name(best_key)):
                best_key = key
    if best_key is None:
        return None
    return best_key, process_to_xsec[best_key]
    


def _merge_values(old_value: Any, new_value: Any) -> Any:
    if old_value is None:
        return copy.deepcopy(new_value)
    if new_value is None:
        return old_value

    if isinstance(old_value, dict) and isinstance(new_value, dict):
        merged = dict(old_value)
        for key, value in new_value.items():
            merged[key] = _merge_values(merged.get(key), value)
        return merged

    if isinstance(old_value, list) and isinstance(new_value, list):
        return old_value + new_value

    if _is_number(old_value) and _is_number(new_value):
        return old_value + new_value

    return copy.deepcopy(new_value)


def _load_single_pkl(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"PKL is not a dictionary: {path}")
    return data


def _load_single_root(path: Path, target_variables: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Load a single ROOT file and convert to pkl-like format."""
    with uproot.open(path) as f:
        if "Events" not in f:
            raise ValueError(f"No 'Events' tree found in ROOT file: {path}")
        
        tree = f["Events"]
        objects = {}

        selected_branches = list(tree.keys())
        if target_variables:
            target_set = set(target_variables)
            selected_branches = [
                branch_name
                for branch_name in tree.keys()
                if branch_name in target_set
                or any(var.startswith(f"{branch_name}_") for var in target_set)
            ]

        # Read selected branches as arrays
        for branch_name in selected_branches:
            try:
                arr = tree[branch_name].array(library="np")
                objects[branch_name] = arr.tolist() if hasattr(arr, 'tolist') else list(arr)
            except Exception as e:
                print(f"Warning: Could not read branch {branch_name}: {e}")
        
        # Get n_events from Metadata tree if available
        n_events = 0
        metadata_n_events_available = False
        if "Metadata" in f:
            meta_tree = f["Metadata"]
            if "n_events" in meta_tree.keys():
                n_events_arr = meta_tree["n_events"].array(library="np")
                # hadd-merged files can carry one n_events entry per source file.
                # Sum all entries so normalization uses the total generated event count.
                n_events = int(np.sum(n_events_arr)) if len(n_events_arr) > 0 else 0
                metadata_n_events_available = n_events > 0

        # Do not fallback to selected entries for MC normalization.
        # If metadata is missing, keep n_events=0 and let caller decide whether to fail.
        return {
            "n_events": n_events,
            "objects": objects,
            "metadata_n_events_available": metadata_n_events_available,
        }


def _should_skip_pkl_file(path: Path) -> bool:
    name = path.name
    return name.endswith(".awk_raw.pkl") or name.endswith("raw.pkl")


def _filter_objects_for_variables(objects: Dict[str, Any], target_variables: Optional[Sequence[str]]) -> Dict[str, Any]:
    if not target_variables:
        return objects
    target_set = set(target_variables)
    filtered: Dict[str, Any] = {}
    for key, value in objects.items():
        if key in target_set or any(var.startswith(f"{key}_") for var in target_set):
            filtered[key] = value
    return filtered


def merge_pkl_folder(folder: Path, target_variables: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Merge all .pkl files in a folder."""
    pkl_files = sorted([p for p in folder.glob("*.pkl") if p.is_file() and not _should_skip_pkl_file(p)])
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in folder: {folder}")

    merged: Dict[str, Any] = {}
    total_files = len(pkl_files)
    loaded_files = 0
    skipped_files = 0
    print(f"Loading {total_files} PKL files from {folder.name}...")
    
    for idx, pkl_path in enumerate(pkl_files, 1):
        try:
            data = _load_single_pkl(pkl_path)
        except Exception as exc:
            skipped_files += 1
            print(f"Warning: Could not load PKL file {pkl_path.name}: {exc}")
            continue
        objects = data.get("objects", {})
        if not isinstance(objects, dict):
            objects = {}
        reduced_data = {
            "n_events": data.get("n_events", 0) or 0,
            "objects": objects,
        }
        merged = _merge_values(merged, reduced_data)
        loaded_files += 1
        
        # Progress indication every 50 files
        if idx % 50 == 0 or idx == total_files:
            print(f"  Progress: {idx}/{total_files} files loaded")

    if loaded_files == 0:
        raise ValueError(f"All PKL files failed to load in folder: {folder}")
    if skipped_files > 0:
        print(f"  Loaded {loaded_files}/{total_files} PKL files (skipped {skipped_files})")

    merged.setdefault("n_events", 0)
    merged.setdefault("objects", {})
    if not isinstance(merged["objects"], dict):
        raise ValueError(f"Merged objects is not a dictionary in folder: {folder}")

    return merged


def merge_root_folder(folder: Path, target_variables: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Merge all .root files in a folder."""
    root_files = sorted([p for p in folder.glob("*.root") if p.is_file()])
    if not root_files:
        raise FileNotFoundError(f"No .root files found in folder: {folder}")

    merged: Dict[str, Any] = {}
    total_files = len(root_files)
    loaded_files = 0
    skipped_files = 0
    print(f"Loading {total_files} ROOT files from {folder.name}...")
    
    for idx, root_path in enumerate(root_files, 1):
        try:
            data = _load_single_root(root_path, target_variables=target_variables)
            if not data.get("metadata_n_events_available", False):
                skipped_files += 1
                print(f"Warning: ROOT file {root_path.name} is missing Metadata.n_events; skipping file.")
                continue
            merged = _merge_values(merged, data)
            loaded_files += 1
            
            # Progress indication every 50 files
            if idx % 50 == 0 or idx == total_files:
                print(f"  Progress: {idx}/{total_files} files loaded")
        except Exception as e:
            print(f"Warning: Could not load ROOT file {root_path.name}: {e}")

    if loaded_files == 0:
        raise ValueError(f"All ROOT files failed to load in folder: {folder}")
    if skipped_files > 0:
        print(f"  Loaded {loaded_files}/{total_files} ROOT files (skipped {skipped_files})")

    merged.setdefault("n_events", 0)
    merged.setdefault("objects", {})
    merged["metadata_n_events_available"] = loaded_files > 0
    if not isinstance(merged["objects"], dict):
        raise ValueError(f"Merged objects is not a dictionary in folder: {folder}")

    return merged


def _flatten_numeric(values: Any) -> np.ndarray:
    result: List[float] = []

    def walk(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, dict):
            return
        if isinstance(item, (list, tuple)):
            for sub in item:
                walk(sub)
            return
        if isinstance(item, np.ndarray):
            for sub in item.ravel().tolist():
                walk(sub)
            return
        if _is_number(item):
            val = float(item)
            if math.isfinite(val):
                result.append(val)

    walk(values)
    if not result:
        return np.array([], dtype=float)
    return np.asarray(result, dtype=float)


def _extract_object_distributions(objects: Dict[str, Any]) -> Dict[str, np.ndarray]:
    distributions: Dict[str, np.ndarray] = {}

    for key, value in objects.items():
        if key.endswith("_mask"):
            continue

        if not isinstance(value, list):
            arr = _flatten_numeric(value)
            if arr.size > 0:
                distributions[key] = arr
            continue

        first_non_none = None
        for item in value:
            if item is not None:
                first_non_none = item
                break

        if first_non_none is None:
            continue

        if isinstance(first_non_none, dict):
            numeric_fields: set[str] = set()
            for item in value:
                if isinstance(item, dict):
                    for field, field_value in item.items():
                        if _is_number(field_value):
                            numeric_fields.add(field)
            for field in sorted(numeric_fields):
                flattened = _flatten_numeric([item.get(field) for item in value if isinstance(item, dict)])
                if flattened.size > 0:
                    distributions[f"{key}_{field}"] = flattened
            continue

        if isinstance(first_non_none, list):
            first_inner = None
            for inner in first_non_none:
                if inner is not None:
                    first_inner = inner
                    break

            if isinstance(first_inner, dict):
                numeric_fields: set[str] = set()
                for event_items in value:
                    if not isinstance(event_items, list):
                        continue
                    for item in event_items:
                        if isinstance(item, dict):
                            for field, field_value in item.items():
                                if _is_number(field_value):
                                    numeric_fields.add(field)

                for field in sorted(numeric_fields):
                    flattened_field: List[float] = []
                    for event_items in value:
                        if not isinstance(event_items, list):
                            continue
                        for item in event_items:
                            if isinstance(item, dict):
                                field_value = item.get(field)
                                if _is_number(field_value):
                                    fv = float(field_value)
                                    if math.isfinite(fv):
                                        flattened_field.append(fv)
                    if flattened_field:
                        distributions[f"{key}_{field}"] = np.asarray(flattened_field, dtype=float)
                continue

            flattened = _flatten_numeric(value)
            if flattened.size > 0:
                distributions[key] = flattened
            continue

        flattened = _flatten_numeric(value)
        if flattened.size > 0:
            distributions[key] = flattened

    return distributions


def _reference_bins_for_variable(variable: Optional[str]) -> Optional[np.ndarray]:
    if not variable:
        return None

    name = variable.lower()
    if "ctsvalue" in name:
        return np.asarray([0.0, 0.25, 0.50, 0.75, 1.0], dtype=float)

    if name.endswith("_pt"):
        return np.asarray([250.0, 300.0, 400.0, 550.0, 1000.0], dtype=float)

    if name in {"pfmet_pt", "met", "recoil"} or name.endswith("met_pt") or "recoil" in name:
        return np.asarray([250.0, 300.0, 400.0, 550.0, 1000.0], dtype=float)

    return None


def _make_bins(all_values: Sequence[np.ndarray], n_bins: int, variable: Optional[str] = None) -> Optional[np.ndarray]:
    reference_bins = _reference_bins_for_variable(variable)
    if reference_bins is not None:
        return reference_bins

    valid_arrays = [arr[np.isfinite(arr)] for arr in all_values if arr.size > 0]
    if not valid_arrays:
        return None

    merged = np.concatenate(valid_arrays)
    if merged.size == 0:
        return None

    data_min = float(np.min(merged))
    data_max = float(np.max(merged))
    if not math.isfinite(data_min) or not math.isfinite(data_max):
        return None

    if abs(data_max - data_min) < 1e-12:
        width = max(1.0, abs(data_max) * 0.05)
        return np.linspace(data_min - width, data_max + width, n_bins + 1)

    is_integer_like = np.allclose(merged, np.round(merged), atol=1e-8)
    unique_count = len(np.unique(np.round(merged).astype(int))) if is_integer_like else 999
    if is_integer_like and unique_count <= 20:
        low = int(np.min(np.round(merged)))
        high = int(np.max(np.round(merged)))
        return np.arange(low - 0.5, high + 1.5, 1.0)

    return np.linspace(data_min, data_max, n_bins + 1)


def _histogram(values: np.ndarray, bins: np.ndarray, n_events: int, luminosity: float = 1.0, cross_section_pb: Optional[float] = None) -> np.ndarray:
    """Create histogram normalized by n_events and scaled by luminosity and cross-section.
    
    Normalization formula:
    - If cross_section_pb is provided: weight = (luminosity × cross_section_fb) / n_events
      where cross_section_fb = cross_section_pb × 1000 (convert pb to fb)
    - If cross_section_pb is None: weight = luminosity / n_events
    """
    if values.size == 0 or n_events <= 0:
        return np.zeros(len(bins) - 1, dtype=float)
    
    if cross_section_pb is not None:
        # Convert cross-section from pb to fb and apply
        cross_section_fb = cross_section_pb * 1000.0
        weight = (luminosity * cross_section_fb) / float(n_events)
    else:
        weight = luminosity / float(n_events)
    
    weights = np.full(values.shape[0], weight, dtype=float)
    hist, _ = np.histogram(values, bins=bins, weights=weights)
    return hist


def _histogram_and_sumw2(
    values: np.ndarray,
    bins: np.ndarray,
    n_events: int,
    luminosity: float = 1.0,
    cross_section_pb: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create weighted histogram and per-bin sumw2 (ROOT-compatible stat unc).

    For a constant event weight w in a given sample, per-bin variance is n_bin * w^2,
    equivalent to ROOT TH1 Sumw2 behavior after scaling.
    """
    if n_events <= 0 or values.size == 0:
        zeros = np.zeros(len(bins) - 1, dtype=float)
        return zeros, zeros

    if cross_section_pb is not None:
        cross_section_fb = cross_section_pb * 1000.0
        weight = (luminosity * cross_section_fb) / float(n_events)
    else:
        weight = luminosity / float(n_events)

    counts, _ = np.histogram(values, bins=bins)
    counts = counts.astype(float)
    hist = counts * weight
    sumw2 = counts * (weight ** 2)
    return hist, sumw2


def _histogram_counts(values: np.ndarray, bins: np.ndarray) -> np.ndarray:
    """Create raw-count histogram without normalization (for data)."""
    if values.size == 0:
        return np.zeros(len(bins) - 1, dtype=float)
    hist, _ = np.histogram(values, bins=bins)
    return hist.astype(float)


def _simplify_sample_label(sample_name: str) -> str:
    label = sample_name
    if label.startswith("new"):
        label = label[3:]
    label = re.sub(r"_20\d{2}(EE)?_.*$", "", label)
    label = re.sub(r"_EVENTSELECTION$", "", label)
    return label


def _get_background_color_map(sample_names: Sequence[str]) -> Dict[str, str]:
    """Map sample names to ROOT-style colors exactly matching CMS reference macro.
    
    ROOT color references from StackPlotter_addMeanWeight.py:
    - DYJets: ROOT.kGreen+1
    - ZJets: ROOT.kAzure-4
    - DIBOSON: ROOT.kBlue+1
    - Top: ROOT.kOrange-1
    - WJets: ROOT.kViolet-2
    - STop: ROOT.kOrange+2
    - GJets: ROOT.kCyan-8
    - QCD: ROOT.kGray+2
    - SMH: ROOT.kRed-1
    """
    # ROOT color to hex mapping (converted from ROOT color codes)
    root_color_map = {
        "DYJets": "#00CC33",        # ROOT.kGreen+1
        "ZJets": "#0066FF",         # ROOT.kAzure-4
        "DIBOSON": "#0000FF",       # ROOT.kBlue+1
        "Top": "#FF6600",           # ROOT.kOrange-1
        "WJets": "#9933FF",         # ROOT.kViolet-2
        "STop": "#FF9900",          # ROOT.kOrange+2
        "GJets": "#00FFFF",         # ROOT.kCyan-8
        "QCD": "#999999",           # ROOT.kGray+2
        "SMH": "#FF0000",           # ROOT.kRed-1
    }

    color_map: Dict[str, str] = {}
    simplified_names = sorted({_simplify_sample_label(name) for name in sample_names})
    for raw_name in simplified_names:
        # Try exact match first
        if raw_name in root_color_map:
            color_map[raw_name] = root_color_map[raw_name]
        # Try substring match for process names
        else:
            matched = False
            for key, color in root_color_map.items():
                if key.lower() in raw_name.lower() or raw_name.lower() in key.lower():
                    color_map[raw_name] = color
                    matched = True
                    break
            # Fallback to default palette if no match
            if not matched:
                fallback_palette = [
                    "#3f90da", "#ffa90e", "#bd1f01", "#94a4a2",
                    "#832db6", "#a96b59", "#e76300", "#b9ac70",
                ]
                color_map[raw_name] = fallback_palette[hash(raw_name) % len(fallback_palette)]
    return color_map


def _variable_unit(variable: str) -> str:
    name = variable.lower()
    if name.endswith("_pt"):
        return "GeV"
    if name.endswith("_phi"):
        return "rad"
    return "arb. unit"


def _apply_variable_plot_filter(variable: str, values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    if variable.lower().endswith("met_pt"):
        return values[values >= 100.0]
    return values


def _plot_stacked_variable(
    variable: str,
    bins: np.ndarray,
    background_hists: List[Tuple[str, np.ndarray, np.ndarray]],
    data_hist: Optional[np.ndarray],
    output_dir: Path,
    luminosity: float = 1.0,
    color_map: Optional[Dict[str, str]] = None,
) -> Path:
    """Produce CMS-style stacked histogram matching ROOT macro exactly.
    
    Key features matching StackPlotter_addMeanWeight.py:
    - ROOT color scheme (kGreen+1, kAzure-4, etc.)
    - MC statistical error band (gray hatched pattern)
    - Data overlaid with black markers (MarkerStyle 20, size 1.3)
    - Log scale with range [0.1, max*1000]
    - Ratio plot (Data/MC) below
    - CMS labels and lumi text at exact positions
    - Legend in 2 columns at [0.50, 0.58, 0.93, 0.92]
    """
    background_hists = [(_simplify_sample_label(label), hist, sumw2) for label, hist, sumw2 in background_hists]
    background_hists = sorted(background_hists, key=lambda item: float(np.sum(item[1])))

    # Set font and axis parameters to match ROOT macro
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["axes.linewidth"] = 1.0

    if color_map is None:
        color_map = _get_background_color_map([label for label, _, _ in background_hists])

    show_ratio = data_hist is not None
    if show_ratio:
        fig, (ax, ax_ratio) = plt.subplots(
            2, 1,
            figsize=(10, 8),
            sharex=True,
            gridspec_kw={"height_ratios": [3.3, 1.0], "hspace": 0.03},
        )
        # Match ROOT pad margins: no ratio pad has bottom=0.09, top=0.08, left=0.12, right=0.06
        fig.subplots_adjust(top=0.94, bottom=0.10, left=0.08, right=0.95)
    else:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax_ratio = None
        fig.subplots_adjust(top=0.94, bottom=0.12, left=0.08, right=0.95)

    # =========================== Main Stack Plot ===========================
    # Stack all background histograms
    cumulative = np.zeros(len(bins) - 1, dtype=float)
    cumulative_sq = np.zeros(len(bins) - 1, dtype=float)  # For MC stat error bars
    
    for label, hist_values, hist_sumw2 in background_hists:
        next_cumulative = cumulative + hist_values
        cumulative_sq += hist_sumw2
        color = color_map.get(label, "#4E79A7")
        ax.stairs(
            next_cumulative,
            bins,
            baseline=cumulative,
            fill=True,
            alpha=1.0,
            linewidth=0.8,
            color=color,
            label=label,
            edgecolor="black",
        )
        cumulative = next_cumulative

    # Draw MC statistical error band (gray hatched pattern)
    mc_stat_err = np.sqrt(cumulative_sq)  # Error = sqrt(sum of squares)
    centers = 0.5 * (bins[:-1] + bins[1:])
    
    # Create error band using fill_between with hatching (mimics ROOT FillStyle 3013)
    # Use centers instead of bins edges to avoid shape mismatch
    ax.fill_between(
        centers,
        cumulative - mc_stat_err,
        cumulative + mc_stat_err,
        alpha=0.4,
        hatch="///",
        facecolor="gray",
        edgecolor="gray",
        linewidth=0.0,
        label="_MC stat",
    )

    # Plot data points with ROOT.TMarkerStyle(20)=circle, size=1.3
    if data_hist is not None:
        half_width = 0.5 * (bins[1:] - bins[:-1])
        mask = data_hist > 0
        if np.any(mask):
            ax.errorbar(
                centers[mask],
                data_hist[mask],
                xerr=half_width[mask],
                yerr=np.sqrt(data_hist[mask]),
                fmt="o",
                color="black",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=4.5,  # ROOT MarkerSize 1.3 -> ~4.5 in matplotlib
                elinewidth=1.2,
                capsize=0,
                label="Data",
                zorder=10,
            )

    # Set axis properties
    ax.set_yscale("log")
    stacked_max = float(np.max(cumulative)) if cumulative.size else 0.0
    data_max = float(np.max(data_hist)) if data_hist is not None and data_hist.size else 0.0
    ymax = max(stacked_max, data_max, 1e-3) * 1000.0
    ax.set_ylim(0.1, ymax)
    ax.set_xlim(float(bins[0]), float(bins[-1]))

    # Y-axis label: "Events/bin" matching ROOT macro
    ax.set_ylabel("Events/bin", fontsize=14)
    
    if show_ratio and ax_ratio is not None:
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
    else:
        ax.set_xlabel(variable, fontsize=14)

    ax.grid(False)
    ax.tick_params(direction="in", which="both", length=6, width=1)

    # =========================== Ratio Plot ===========================
    if show_ratio and ax_ratio is not None and data_hist is not None:
        pred = cumulative
        # ROOT reference uses (Data-Pred)/Pred, not Data/MC.
        ratio = np.divide(
            data_hist - pred,
            pred,
            out=np.full_like(data_hist, np.nan, dtype=float),
            where=pred > 0,
        )
        ratio_err = np.divide(
            np.sqrt(data_hist),
            pred,
            out=np.zeros_like(data_hist, dtype=float),
            where=pred > 0,
        )
        ratio_mask = np.isfinite(ratio)

        # Horizontal line at 0 for (Data-Pred)/Pred.
        ax_ratio.axhline(0.0, color="black", linestyle="-", linewidth=1.2)
        
        # Data/MC points
        if np.any(ratio_mask):
            half_width = 0.5 * (bins[1:] - bins[:-1])
            ax_ratio.errorbar(
                centers[ratio_mask],
                ratio[ratio_mask],
                yerr=ratio_err[ratio_mask],
                fmt="o",
                color="black",
                markerfacecolor="black",
                markeredgecolor="black",
                markersize=4.5,
                elinewidth=1.0,
                capsize=0,
                zorder=10,
            )

        # Ratio error band (MC stat errors)
        pred_rel_err = np.divide(
            mc_stat_err,
            pred,
            out=np.zeros_like(pred, dtype=float),
            where=pred > 0,
        )
        ax_ratio.fill_between(
            centers,
            -pred_rel_err,
            pred_rel_err,
            alpha=0.3,
            hatch="///",
            facecolor="gray",
            edgecolor="gray",
            linewidth=0.0,
        )

        ax_ratio.set_ylabel("(Data-Pred)/Pred", fontsize=12)
        ax_ratio.set_xlabel(variable, fontsize=14)
        if "qcd" in variable.lower():
            ax_ratio.set_ylim(-1.18, 1.18)
        else:
            ax_ratio.set_ylim(-0.68, 0.68)
        ax_ratio.set_xlim(float(bins[0]), float(bins[-1]))
        ax_ratio.grid(False)
        ax_ratio.tick_params(direction="in", which="both", length=6, width=1)

    # =========================== CMS Labels ===========================
    # Match TPaveText positions from ROOT macro exactly
    fig.text(0.114, 0.955, "CMS", fontsize=18, fontweight="bold", ha="left", va="top")
    fig.text(0.218, 0.946, "Work in progress", fontsize=12, style="italic", ha="left", va="top")
    fig.text(0.680, 0.955, f"{luminosity:.2f} fb$^{{-1}}$ (13 TeV)", fontsize=12, ha="left", va="top")

    # =========================== Legend ===========================
    # ROOT macro: SetLegend([.50, .58, .93, .92], ncol=2)
    # Matplotlib bbox_to_anchor: (x, y, width, height)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Filter out the "_MC stat" dummy entry
        filtered_handles = []
        filtered_labels = []
        for h, l in zip(handles, labels):
            if "_MC stat" not in l:
                filtered_handles.append(h)
                filtered_labels.append(l)
        
        if filtered_handles:
            legend = ax.legend(
                filtered_handles,
                filtered_labels,
                loc="upper left",
                bbox_to_anchor=(0.50, 0.58, 0.43, 0.34),
                mode="expand",
                ncol=2,
                frameon=False,
                borderaxespad=0.0,
                handlelength=1.5,
                columnspacing=1.0,
                handletextpad=0.4,
                fontsize=11,
            )

    # Set spine linewidths
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"stacked_{variable}.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _load_sample_from_folder(
    folder: Path,
    luminosity: float = 1.0,
    file_type: Literal["auto", "pkl", "root"] = "auto",
    target_variables: Optional[Sequence[str]] = None,
    xsection_data: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    year: Optional[str] = None,
    is_data: bool = False,
) -> SampleData:
    """Load sample from folder with configurable file-type selection."""
    root_files = [p for p in folder.glob("*.root") if p.is_file()]
    pkl_files = [p for p in folder.glob("*.pkl") if p.is_file() and not _should_skip_pkl_file(p)]

    if file_type == "pkl":
        if not pkl_files:
            raise FileNotFoundError(f"No eligible .pkl files found in folder: {folder}")
        merged = merge_pkl_folder(folder, target_variables=target_variables)
    elif file_type == "root":
        if not root_files:
            raise FileNotFoundError(f"No .root files found in folder: {folder}")
        merged = merge_root_folder(folder, target_variables=target_variables)
    else:
        if root_files:
            merged = merge_root_folder(folder, target_variables=target_variables)
        elif pkl_files:
            merged = merge_pkl_folder(folder, target_variables=target_variables)
        else:
            raise FileNotFoundError(f"No .pkl or .root files found in folder: {folder}")
    
    n_events = int(merged.get("n_events", 0) or 0)
    objects = merged.get("objects", {})
    metadata_n_events_available = bool(merged.get("metadata_n_events_available", True))
    if not isinstance(objects, dict):
        raise ValueError(f"objects is not a dictionary for folder: {folder}")

    # Build per-process sub-samples for ROOT inputs so each pt bin uses its own cross-section.
    sub_samples: Optional[List[SubSampleData]] = None
    if root_files and xsection_data and year and not is_data:
        process_to_xsec = _build_xsection_process_lookup(xsection_data, year)
        per_process_merged: Dict[str, Dict[str, Any]] = {}
        unmatched_files: List[str] = []
        unmatched_merged: Dict[str, Any] = {
            "n_events": 0,
            "objects": {},
            "metadata_n_events_available": True,
            "cross_section": None,
        }

        for root_file in sorted(root_files):
            try:
                loaded = _load_single_root(root_file, target_variables=target_variables)
            except Exception as exc:
                print(f"Warning: Could not load ROOT file {root_file.name}: {exc}")
                continue

            if not loaded.get("metadata_n_events_available", False):
                print(
                    f"Warning: ROOT file {root_file.name} is missing Metadata.n_events; skipping for xsection grouping."
                )
                continue

            match = _match_root_file_to_process(root_file.name, process_to_xsec)
            if match is None:
                unmatched_files.append(root_file.name)
                unmatched_merged["n_events"] = int(unmatched_merged.get("n_events", 0) or 0) + int(
                    loaded.get("n_events", 0) or 0
                )
                unmatched_merged["objects"] = _merge_values(unmatched_merged.get("objects", {}), loaded.get("objects", {}))
                continue
            process_key, xsec_pb = match
            group = per_process_merged.get(process_key)
            if group is None:
                group = {
                    "n_events": 0,
                    "objects": {},
                    "metadata_n_events_available": True,
                    "cross_section": xsec_pb,
                }
                per_process_merged[process_key] = group
            group["n_events"] = int(group.get("n_events", 0) or 0) + int(loaded.get("n_events", 0) or 0)
            group["objects"] = _merge_values(group.get("objects", {}), loaded.get("objects", {}))

        if unmatched_files:
            if per_process_merged:
                print(
                    f"Warning: Could not match {len(unmatched_files)} ROOT files in {folder.name} "
                    f"to JSON full_dataset keys; unmatched files will use fallback normalization. "
                    f"Example: {unmatched_files[0]}"
                )
            else:
                print(
                    f"Warning: No ROOT files in {folder.name} matched JSON full_dataset keys; "
                    "falling back to folder-level cross-section matching."
                )

        sub_samples = []
        print(f"Process-xsection mapping for {folder.name} ({year}):")
        for process_key in sorted(per_process_merged.keys()):
            group = per_process_merged[process_key]
            group_n_events = int(group.get("n_events", 0) or 0)
            group_xsec = float(group.get("cross_section", 0.0))
            print(f"  {process_key} -> xsec={group_xsec:.6g} pb, n_events={group_n_events}")
            if group_n_events <= 0:
                raise ValueError(
                    f"Invalid n_events={group_n_events} for sub-sample {process_key} in {folder.name}."
                )
            if not bool(group.get("metadata_n_events_available", True)):
                raise ValueError(
                    f"Sub-sample {process_key} in {folder.name} has ROOT file(s) without Metadata.n_events."
                )
            sub_samples.append(
                SubSampleData(
                    process=process_key,
                    n_events=group_n_events,
                    objects=group.get("objects", {}),
                    cross_section=group_xsec,
                )
            )
        unmatched_n_events = int(unmatched_merged.get("n_events", 0) or 0)
        if unmatched_n_events > 0:
            print(
                f"  __unmatched__ -> xsec=None (fallback), n_events={unmatched_n_events}"
            )
            sub_samples.append(
                SubSampleData(
                    process="__unmatched__",
                    n_events=unmatched_n_events,
                    objects=unmatched_merged.get("objects", {}),
                    cross_section=None,
                )
            )
    
    # Extract folder-level cross-section only when per-process matching is unavailable.
    # This avoids confusing logs for mixed-pt folders that already use sub-sample xsections.
    cross_section_pb = None
    if xsection_data and year and not sub_samples:
        cross_section_pb = _extract_xsection_by_year(xsection_data, folder.name, year)
        if cross_section_pb is not None:
            print(f"Found cross-section for {folder.name} ({year}): {cross_section_pb:.4f} pb")

    # Enforce strict MC normalization:
    # 1) n_events must come from metadata for ROOT-based samples
    # 2) when xsection mode is enabled, MC sample must have a matched cross-section
    if not is_data:
        if xsection_data and year and cross_section_pb is None and not sub_samples:
            raise ValueError(
                f"No cross-section found for MC sample {folder.name} in year {year}. "
                "Cannot apply strict MC normalization."
            )
        if n_events <= 0:
            raise ValueError(
                f"Invalid n_events={n_events} for MC sample {folder.name}. "
                "Expected positive total generated events from metadata."
            )
    
    return SampleData(
        name=folder.name,
        n_events=n_events,
        objects=objects,
        luminosity=luminosity,
        cross_section=cross_section_pb,
        metadata_n_events_available=metadata_n_events_available,
        sub_samples=sub_samples,
    )


def run_plotting(
    data_folder: Optional[Path],
    data_folders: Optional[Sequence[Path]],
    background_folders: Sequence[Path],
    output_dir: Path,
    variables: Optional[Sequence[str]] = None,
    n_bins: int = 40,
    max_variables: Optional[int] = None,
    luminosity: float = 1.0,
    file_type: Literal["auto", "pkl", "root"] = "auto",
    xsection_json_path: Optional[Path] = None,
    year: Optional[str] = None,
) -> List[Path]:
    if not background_folders:
        raise ValueError("At least one background folder is required")

    # Load cross-section data if provided
    xsection_data = None
    if xsection_json_path:
        xsection_data = _load_xsection_json(xsection_json_path)
        print(f"Loaded cross-section data from: {xsection_json_path}")

    bkg_samples = [
        _load_sample_from_folder(
            folder,
            luminosity,
            file_type=file_type,
            target_variables=variables,
            xsection_data=xsection_data,
            year=year,
            is_data=False,
        )
        for folder in background_folders
    ]
    combined_data_folders: List[Path] = []
    if data_folders:
        combined_data_folders.extend(data_folders)
    if data_folder is not None:
        combined_data_folders.append(data_folder)

    data_sample: Optional[SampleData] = None
    if combined_data_folders:
        merged_data: Dict[str, Any] = {"n_events": 0, "objects": {}}
        for folder in combined_data_folders:
            loaded_data = _load_sample_from_folder(
                folder,
                luminosity,
                file_type=file_type,
                target_variables=variables,
                xsection_data=xsection_data,
                year=year,
                is_data=True,
            )
            merged_data["n_events"] = int(merged_data.get("n_events", 0) or 0) + int(loaded_data.n_events)
            merged_data["objects"] = _merge_values(merged_data.get("objects", {}), loaded_data.objects)

        data_sample = SampleData(
            name="data",
            n_events=int(merged_data.get("n_events", 0) or 0),
            objects=merged_data.get("objects", {}),
            luminosity=luminosity,
        )

    bkg_dist_by_sample = {sample.name: _extract_object_distributions(sample.objects) for sample in bkg_samples}
    data_dist = _extract_object_distributions(data_sample.objects) if data_sample else {}

    print("MC normalization summary:")
    for sample in bkg_samples:
        if sample.sub_samples:
            matched = sum(1 for sub in sample.sub_samples if sub.cross_section is not None)
            unmatched = len(sample.sub_samples) - matched
            print(
                f"  {sample.name}: n_events={sample.n_events}, lumi={sample.luminosity} fb^-1, "
                f"xsec=per-process ({matched} matched"
                + (f", {unmatched} fallback" if unmatched > 0 else "")
                + ")"
            )
        elif sample.cross_section is not None:
            event_weight = (sample.luminosity * sample.cross_section * 1000.0) / float(sample.n_events)
            print(
                f"  {sample.name}: n_events={sample.n_events}, lumi={sample.luminosity} fb^-1, "
                f"xsec={sample.cross_section:.6g} pb, event_weight={event_weight:.6g}"
            )
        else:
            event_weight = sample.luminosity / float(sample.n_events) if sample.n_events > 0 else 0.0
            print(
                f"  {sample.name}: n_events={sample.n_events}, lumi={sample.luminosity} fb^-1, "
                f"xsec=None, fallback_event_weight={event_weight:.6g}"
            )

    variable_names: set[str] = set()
    for dist in bkg_dist_by_sample.values():
        variable_names.update(dist.keys())
    variable_names.update(data_dist.keys())

    if variables:
        selected_vars = [v for v in variables if v in variable_names]
    else:
        selected_vars = sorted(variable_names)

    if max_variables is not None:
        selected_vars = selected_vars[:max_variables]

    color_map = _get_background_color_map([sample.name for sample in bkg_samples])

    created_files: List[Path] = []
    for variable in selected_vars:
        all_values = []
        for dist in bkg_dist_by_sample.values():
            if variable in dist:
                all_values.append(_apply_variable_plot_filter(variable, dist[variable]))
        if variable in data_dist:
            all_values.append(_apply_variable_plot_filter(variable, data_dist[variable]))

        bins = _make_bins(all_values, n_bins=n_bins, variable=variable)
        if bins is None or len(bins) < 2:
            continue

        bkg_hists: List[Tuple[str, np.ndarray, np.ndarray]] = []
        for sample in bkg_samples:
            if sample.sub_samples:
                # Mixed-pt folders are summed from per-process components, each with its own xsec.
                hist_values = np.zeros(len(bins) - 1, dtype=float)
                hist_sumw2 = np.zeros(len(bins) - 1, dtype=float)
                for sub in sample.sub_samples:
                    sub_dist = _extract_object_distributions(sub.objects)
                    values = sub_dist.get(variable, np.array([], dtype=float))
                    values = _apply_variable_plot_filter(variable, values)
                    sub_hist, sub_sumw2 = _histogram_and_sumw2(
                        values,
                        bins,
                        sub.n_events,
                        sample.luminosity,
                        cross_section_pb=sub.cross_section,
                    )
                    hist_values += sub_hist
                    hist_sumw2 += sub_sumw2
            else:
                sample_dist = bkg_dist_by_sample.get(sample.name, {})
                values = sample_dist.get(variable, np.array([], dtype=float))
                values = _apply_variable_plot_filter(variable, values)
                hist_values, hist_sumw2 = _histogram_and_sumw2(
                    values,
                    bins,
                    sample.n_events,
                    sample.luminosity,
                    cross_section_pb=sample.cross_section,
                )
            bkg_hists.append((sample.name, hist_values, hist_sumw2))

        if np.allclose(sum(hist for _, hist, _ in bkg_hists), 0.0):
            continue

        data_hist = None
        if data_sample is not None:
            values = data_dist.get(variable, np.array([], dtype=float))
            values = _apply_variable_plot_filter(variable, values)
            data_hist = _histogram_counts(values, bins)

        out_path = _plot_stacked_variable(
            variable,
            bins,
            bkg_hists,
            data_hist,
            output_dir,
            luminosity,
            color_map=color_map,
        )
        created_files.append(out_path)

    return created_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stacked plots from event-level PKL/ROOT folders."
    )
    parser.add_argument("--data-folder", type=Path, default=None, help="(Optional) Folder containing data PKL/ROOT file(s). If not provided, no data overlay will be shown.")
    parser.add_argument(
        "--data-folders",
        type=Path,
        nargs="+",
        default=None,
        help="(Optional) Multiple data folders to merge before plotting.",
    )
    parser.add_argument(
        "--background-folders",
        type=Path,
        nargs="+",
        required=True,
        help="One or more background folders. Each folder is merged into one stack component.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for output stacked plot PNG files")
    parser.add_argument("--variables", nargs="+", default=None, help="Optional list of variables to plot")
    parser.add_argument("--bins", type=int, default=40, help="Number of bins for continuous variables")
    parser.add_argument("--max-variables", type=int, default=None, help="Optional maximum number of variables to plot")
    parser.add_argument("--lumi", type=float, default=1.0, help="Luminosity in fb^-1 for normalization (default: 1.0)")
    parser.add_argument(
        "--file-type",
        choices=["auto", "pkl", "root"],
        default="auto",
        help="Input file type selection for each sample folder (default: auto).",
    )
    parser.add_argument(
        "--xsection-json",
        type=Path,
        default=None,
        help="(Optional) Path to JSON file containing cross-section data for samples.",
    )
    parser.add_argument(
        "--year",
        type=str,
        default=None,
        help="(Optional) Year to use for cross-section lookup (e.g., 2022, 2022EE, 2023).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_paths = run_plotting(
        data_folder=args.data_folder,
        data_folders=args.data_folders,
        background_folders=args.background_folders,
        output_dir=args.output_dir,
        variables=args.variables,
        n_bins=args.bins,
        max_variables=args.max_variables,
        luminosity=args.lumi,
        file_type=args.file_type,
        xsection_json_path=args.xsection_json,
        year=args.year,
    )
    print(f"Created {len(output_paths)} stacked plot(s)")
    data_inputs: List[Path] = []
    if args.data_folders:
        data_inputs.extend(args.data_folders)
    if args.data_folder:
        data_inputs.append(args.data_folder)
    if data_inputs:
        if len(data_inputs) == 1:
            print(f"Data folder: {data_inputs[0]}")
        else:
            print("Data folders:")
            for folder in data_inputs:
                print(f"  - {folder}")
    else:
        print("No data folder provided - plotting backgrounds only")
    print(f"Luminosity: {args.lumi} fb^-1")
    print(f"File type: {args.file_type}")
    for out in output_paths:
        print(out)


if __name__ == "__main__":
    main()
