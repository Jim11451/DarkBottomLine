#!/usr/bin/env python3
"""
Standalone stacked plotting tool for event-level output PKL files.

Features:
- Reads new event-level PKL format (keys: n_events/events/objects)
- Merges multiple PKL files inside each input folder
- Supports multiple background folders and one data folder
- Draws log-scale stacked background plot with data overlay
- Normalizes all histograms by n_events
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SampleData:
    name: str
    n_events: int
    objects: Dict[str, Any]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


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


def merge_pkl_folder(folder: Path) -> Dict[str, Any]:
    pkl_files = sorted([p for p in folder.glob("*.pkl") if p.is_file() and not p.name.endswith(".awk_raw.pkl")])
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in folder: {folder}")

    merged: Dict[str, Any] = {}
    for pkl_path in pkl_files:
        data = _load_single_pkl(pkl_path)
        merged = _merge_values(merged, data)

    merged.setdefault("n_events", 0)
    merged.setdefault("objects", {})
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


def _make_bins(all_values: Sequence[np.ndarray], n_bins: int) -> Optional[np.ndarray]:
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


def _histogram(values: np.ndarray, bins: np.ndarray, n_events: int) -> np.ndarray:
    if values.size == 0 or n_events <= 0:
        return np.zeros(len(bins) - 1, dtype=float)
    weights = np.full(values.shape[0], 1.0 / float(n_events), dtype=float)
    hist, _ = np.histogram(values, bins=bins, weights=weights)
    return hist


def _plot_stacked_variable(
    variable: str,
    bins: np.ndarray,
    background_hists: List[Tuple[str, np.ndarray]],
    data_hist: Optional[np.ndarray],
    output_dir: Path,
) -> Path:
    background_hists = sorted(background_hists, key=lambda item: float(np.sum(item[1])))

    color_cycle = plt.cm.tab20.colors
    fig, ax = plt.subplots(figsize=(9, 7))

    cumulative = np.zeros(len(bins) - 1, dtype=float)
    for idx, (label, hist_values) in enumerate(background_hists):
        next_cumulative = cumulative + hist_values
        color = color_cycle[idx % len(color_cycle)]
        ax.stairs(next_cumulative, bins, baseline=cumulative, fill=True, alpha=0.75, linewidth=1.0, color=color, label=label)
        cumulative = next_cumulative

    if data_hist is not None:
        ax.stairs(data_hist, bins, color="black", linewidth=2.0, label="Data", fill=False)

    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-6)
    ax.set_xlabel(variable)
    ax.set_ylabel("Entries / n_events")
    ax.set_title(f"Stacked plot: {variable}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"stacked_{variable}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _load_sample_from_folder(folder: Path) -> SampleData:
    merged = merge_pkl_folder(folder)
    n_events = int(merged.get("n_events", 0) or 0)
    objects = merged.get("objects", {})
    if not isinstance(objects, dict):
        raise ValueError(f"objects is not a dictionary for folder: {folder}")
    return SampleData(name=folder.name, n_events=n_events, objects=objects)


def run_plotting(
    data_folder: Optional[Path],
    background_folders: Sequence[Path],
    output_dir: Path,
    variables: Optional[Sequence[str]] = None,
    n_bins: int = 40,
    max_variables: Optional[int] = None,
) -> List[Path]:
    if not background_folders:
        raise ValueError("At least one background folder is required")

    bkg_samples = [_load_sample_from_folder(folder) for folder in background_folders]
    data_sample = _load_sample_from_folder(data_folder) if data_folder else None

    bkg_dist_by_sample = {sample.name: _extract_object_distributions(sample.objects) for sample in bkg_samples}
    data_dist = _extract_object_distributions(data_sample.objects) if data_sample else {}

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

    created_files: List[Path] = []
    for variable in selected_vars:
        all_values = []
        for dist in bkg_dist_by_sample.values():
            if variable in dist:
                all_values.append(dist[variable])
        if variable in data_dist:
            all_values.append(data_dist[variable])

        bins = _make_bins(all_values, n_bins=n_bins)
        if bins is None or len(bins) < 2:
            continue

        bkg_hists: List[Tuple[str, np.ndarray]] = []
        for sample in bkg_samples:
            sample_dist = bkg_dist_by_sample.get(sample.name, {})
            values = sample_dist.get(variable, np.array([], dtype=float))
            hist_values = _histogram(values, bins, sample.n_events)
            bkg_hists.append((sample.name, hist_values))

        if np.allclose(sum(hist for _, hist in bkg_hists), 0.0):
            continue

        data_hist = None
        if data_sample is not None:
            values = data_dist.get(variable, np.array([], dtype=float))
            data_hist = _histogram(values, bins, data_sample.n_events)

        out_path = _plot_stacked_variable(variable, bins, bkg_hists, data_hist, output_dir)
        created_files.append(out_path)

    return created_files


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create stacked plots from event-level PKL folders (new output format)."
    )
    parser.add_argument("--data-folder", type=Path, default=None, help="Folder containing data PKL file(s)")
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_paths = run_plotting(
        data_folder=args.data_folder,
        background_folders=args.background_folders,
        output_dir=args.output_dir,
        variables=args.variables,
        n_bins=args.bins,
        max_variables=args.max_variables,
    )
    print(f"Created {len(output_paths)} stacked plot(s)")
    for out in output_paths:
        print(out)


if __name__ == "__main__":
    main()