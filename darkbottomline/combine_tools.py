"""
Higgs Combine integration tools for DarkBottomLine framework.
"""

import subprocess
import json
import logging
import yaml
import numpy as np
import awkward as ak
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import uproot
# import ROOT  # Optional: ROOT not available


class CombineDatacardWriter:
    """
    Generates Combine-compatible datacards and workspaces.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize datacard writer.

        Args:
            config: Combine configuration dictionary
        """
        self.config = config
        self.datacard_config = config.get("datacard", {})
        self.output_config = config.get("output", {})
        self.advanced_config = config.get("advanced", {})

        logging.info("CombineDatacardWriter initialized")

    def write_datacard(self, results: Dict[str, Any], output_dir: str,
                      region: str = "SR", year: str = "2023") -> str:
        """
        Write Combine datacard.

        Args:
            results: Analysis results dictionary
            output_dir: Output directory
            region: Analysis region
            year: Data-taking year

        Returns:
            Path to generated datacard file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        datacard_file = output_path / self.output_config.get("datacard_file", "datacard.txt")

        # Generate datacard content
        datacard_content = self._generate_datacard_content(results, region, year)

        # Write datacard file
        with open(datacard_file, 'w') as f:
            f.write(datacard_content)

        logging.info(f"Datacard written to {datacard_file}")

        return str(datacard_file)

    def _generate_datacard_content(self, results: Dict[str, Any], region: str, year: str) -> str:
        """
        Generate datacard content.

        Args:
            results: Analysis results
            region: Analysis region
            year: Data-taking year

        Returns:
            Datacard content string
        """
        lines = []

        # Header
        lines.append(f"# Datacard for DarkBottomLine {region} analysis ({year})")
        lines.append(f"# Generated automatically by DarkBottomLine framework")
        lines.append("")

        # Imax (number of bins)
        n_bins = self._get_number_of_bins(results, region)
        lines.append(f"imax {n_bins} number of bins")

        # Jmax (number of processes - 1)
        processes = list(self.datacard_config.get("processes", {}).keys())
        n_processes = len(processes)
        lines.append(f"jmax {n_processes - 1} number of processes minus 1")

        # Kmax (number of nuisance parameters)
        n_nuisance = len(self.datacard_config.get("systematics", {}))
        lines.append(f"kmax {n_nuisance} number of nuisance parameters")
        lines.append("")

        # Bin names
        lines.append("# Bin names")
        lines.append("bin".ljust(20) + " ".join([region] * n_bins))
        lines.append("")

        # Observation
        lines.append("# Observation")
        obs_values = self._get_observation_values(results, region)
        lines.append("observation".ljust(20) + " ".join([str(int(v)) for v in obs_values]))
        lines.append("")

        # Process names
        lines.append("# Process names")
        process_line = "bin".ljust(20)
        for _ in range(n_bins):
            for process in processes:
                process_line += f" {region}"
        lines.append(process_line)

        process_line = "process".ljust(20)
        for _ in range(n_bins):
            for i, process in enumerate(processes):
                process_line += f" {i}"
        lines.append(process_line)

        process_line = "process".ljust(20)
        for _ in range(n_bins):
            for process in processes:
                process_line += f" {process}"
        lines.append(process_line)

        # Rates
        lines.append("# Rates")
        rate_line = "rate".ljust(20)
        for _ in range(n_bins):
            for process in processes:
                rate = self._get_process_rate(results, process, region)
                rate_line += f" {rate:.6f}"
        lines.append(rate_line)
        lines.append("")

        # Systematic uncertainties
        lines.append("# Systematic uncertainties")
        for sys_name, sys_config in self.datacard_config.get("systematics", {}).items():
            sys_type = sys_config.get("type", "lnN")
            sys_processes = sys_config.get("processes", [])
            sys_value = sys_config.get("value", 1.0)

            if sys_type == "lnN":
                sys_line = f"{sys_name}".ljust(20)
                for _ in range(n_bins):
                    for process in processes:
                        if process in sys_processes:
                            sys_line += f" {sys_value:.3f}"
                        else:
                            sys_line += " -"
                lines.append(sys_line)
            elif sys_type == "shape":
                sys_line = f"{sys_name} shape".ljust(20)
                for _ in range(n_bins):
                    for process in processes:
                        if process in sys_processes:
                            sys_line += f" {sys_value:.3f}"
                        else:
                            sys_line += " -"
                lines.append(sys_line)

        lines.append("")

        # Rate parameters
        lines.append("# Rate parameters")
        for rate_name, rate_config in self.datacard_config.get("rate_parameters", {}).items():
            rate_type = rate_config.get("type", "rateParam")
            rate_regions = rate_config.get("regions", [])
            rate_processes = rate_config.get("processes", [])
            rate_value = rate_config.get("value", 1.0)
            rate_uncertainty = rate_config.get("uncertainty", 0.1)

            if rate_type == "rateParam":
                rate_line = f"{rate_name} rateParam {region} {','.join(rate_processes)} {rate_value:.3f} [{rate_uncertainty:.3f}]"
                lines.append(rate_line)

        return "\n".join(lines)

    def _get_number_of_bins(self, results: Dict[str, Any], region: str) -> int:
        """Get number of bins for the region."""
        # Placeholder implementation
        return 1

    def _get_observation_values(self, results: Dict[str, Any], region: str) -> List[float]:
        """Get observation values for the region."""
        # Placeholder implementation
        return [100.0]  # Example observation

    def _get_process_rate(self, results: Dict[str, Any], process: str, region: str) -> float:
        """Get process rate for the region."""
        # Placeholder implementation
        rates = {
            "signal": 10.0,
            "ttbar": 50.0,
            "wjets": 30.0,
            "zjets": 20.0,
            "qcd": 100.0,
            "st": 15.0,
            "diboson": 5.0
        }
        return rates.get(process, 1.0)

    def write_shapes(self, results: Dict[str, Any], output_dir: str,
                    region: str = "SR") -> str:
        """
        Write ROOT shapes file.

        Args:
            results: Analysis results dictionary
            output_dir: Output directory
            region: Analysis region

        Returns:
            Path to generated shapes file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        shapes_file = output_path / self.output_config.get("shapes_file", "shapes.root")

        # Create ROOT file with histograms
        with uproot.recreate(str(shapes_file)) as f:
            # Write histograms for each process
            for process in self.datacard_config.get("processes", {}).keys():
                # Create placeholder histogram
                hist_name = f"{region}_{process}"
                hist_data = self._create_histogram_data(results, process, region)

                # Write histogram to ROOT file
                f[hist_name] = hist_data

        logging.info(f"Shapes file written to {shapes_file}")

        return str(shapes_file)

    def _create_histogram_data(self, results: Dict[str, Any], process: str, region: str) -> np.ndarray:
        """Create histogram data for a process."""
        # Placeholder implementation
        return np.random.poisson(50, 10)  # Example histogram data

    def create_workspace(self, datacard_file: str, output_dir: str) -> str:
        """
        Create Combine workspace from datacard.

        Args:
            datacard_file: Path to datacard file
            output_dir: Output directory

        Returns:
            Path to generated workspace file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        workspace_file = output_path / self.output_config.get("workspace_file", "workspace.root")

        # Run text2workspace
        cmd = [
            self.advanced_config.get("combine_commands", {}).get("text2workspace", "text2workspace.py"),
            str(datacard_file),
            "-o", str(workspace_file)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logging.info(f"Workspace created: {workspace_file}")
            logging.debug(f"text2workspace output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to create workspace: {e}")
            logging.error(f"Error output: {e.stderr}")
            raise

        return str(workspace_file)


class CombineRunner:
    """
    Runs Combine fits and analyses.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Combine runner.

        Args:
            config: Combine configuration dictionary
        """
        self.config = config
        self.fit_config = config.get("fit", {})
        self.output_config = config.get("output", {})
        self.advanced_config = config.get("advanced", {})

        logging.info("CombineRunner initialized")

    def run_asymptotic_limits(self, datacard_file: str, output_dir: str,
                             options: Optional[Dict[str, Any]] = None) -> str:
        """
        Run AsymptoticLimits.

        Args:
            datacard_file: Path to datacard file
            output_dir: Output directory
            options: Additional options

        Returns:
            Path to results file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / self.output_config.get("fit_results", {}).get("asymptotic_limits", "asymptotic_limits.root")

        # Build command
        cmd = [
            self.advanced_config.get("combine_commands", {}).get("combine", "combine"),
            str(datacard_file),
            "-M", "AsymptoticLimits"
        ]

        # Add options
        cmd.extend(self.advanced_config.get("combine_options", {}).get("asymptotic_limits", []))

        if options:
            for key, value in options.items():
                cmd.extend([f"--{key}", str(value)])

        # Run command
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(output_path))
            logging.info(f"AsymptoticLimits completed: {results_file}")
            logging.debug(f"Combine output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"AsymptoticLimits failed: {e}")
            logging.error(f"Error output: {e.stderr}")
            raise

        return str(results_file)

    def run_fit_diagnostics(self, datacard_file: str, output_dir: str,
                           fit_region: str = "SR", include_signal: bool = False,
                           options: Optional[Dict[str, Any]] = None) -> str:
        """
        Run FitDiagnostics.

        Args:
            datacard_file: Path to datacard file
            output_dir: Output directory
            fit_region: Region to fit
            include_signal: Whether to include signal
            options: Additional options

        Returns:
            Path to results file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / self.output_config.get("fit_results", {}).get("fit_diagnostics", "fitDiagnostics.root")

        # Build command
        cmd = [
            self.advanced_config.get("combine_commands", {}).get("combine", "combine"),
            str(datacard_file),
            "-M", "FitDiagnostics"
        ]

        # Add options
        cmd.extend(self.advanced_config.get("combine_options", {}).get("fit_diagnostics", []))

        if not include_signal:
            cmd.extend(["--setParameters", "r=0"])

        if options:
            for key, value in options.items():
                cmd.extend([f"--{key}", str(value)])

        # Run command
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(output_path))
            logging.info(f"FitDiagnostics completed: {results_file}")
            logging.debug(f"Combine output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"FitDiagnostics failed: {e}")
            logging.error(f"Error output: {e.stderr}")
            raise

        return str(results_file)

    def run_goodness_of_fit(self, datacard_file: str, output_dir: str,
                           toys: int = 1000, options: Optional[Dict[str, Any]] = None) -> str:
        """
        Run GoodnessOfFit.

        Args:
            datacard_file: Path to datacard file
            output_dir: Output directory
            toys: Number of toys
            options: Additional options

        Returns:
            Path to results file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / self.output_config.get("fit_results", {}).get("goodness_of_fit", "gof.root")

        # Build command
        cmd = [
            self.advanced_config.get("combine_commands", {}).get("combine", "combine"),
            str(datacard_file),
            "-M", "GoodnessOfFit"
        ]

        # Add options
        cmd.extend(self.advanced_config.get("combine_options", {}).get("goodness_of_fit", []))
        cmd.extend(["--toys", str(toys)])

        if options:
            for key, value in options.items():
                cmd.extend([f"--{key}", str(value)])

        # Run command
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(output_path))
            logging.info(f"GoodnessOfFit completed: {results_file}")
            logging.debug(f"Combine output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"GoodnessOfFit failed: {e}")
            logging.error(f"Error output: {e.stderr}")
            raise

        return str(results_file)

    def run_impacts(self, datacard_file: str, output_dir: str,
                   options: Optional[Dict[str, Any]] = None) -> str:
        """
        Run impacts calculation.

        Args:
            datacard_file: Path to datacard file
            output_dir: Output directory
            options: Additional options

        Returns:
            Path to results file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results_file = output_path / self.output_config.get("fit_results", {}).get("impacts", "impacts.json")

        # Build command
        cmd = [
            self.advanced_config.get("combine_commands", {}).get("combine_tool", "combineTool.py"),
            "combine",
            "-M", "Impacts",
            "-d", str(datacard_file)
        ]

        # Add options
        cmd.extend(self.advanced_config.get("combine_options", {}).get("impacts", []))

        if options:
            for key, value in options.items():
                cmd.extend([f"--{key}", str(value)])

        # Run command
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=str(output_path))
            logging.info(f"Impacts calculation completed: {results_file}")
            logging.debug(f"CombineTool output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Impacts calculation failed: {e}")
            logging.error(f"Error output: {e.stderr}")
            raise

        return str(results_file)

    def parse_results(self, results_file: str, mode: str) -> Dict[str, Any]:
        """
        Parse Combine results.

        Args:
            results_file: Path to results file
            mode: Combine mode (AsymptoticLimits, FitDiagnostics, etc.)

        Returns:
            Parsed results dictionary
        """
        results = {}

        if mode == "AsymptoticLimits":
            results = self._parse_asymptotic_limits(results_file)
        elif mode == "FitDiagnostics":
            results = self._parse_fit_diagnostics(results_file)
        elif mode == "GoodnessOfFit":
            results = self._parse_goodness_of_fit(results_file)
        elif mode == "Impacts":
            results = self._parse_impacts(results_file)

        return results

    def _parse_asymptotic_limits(self, results_file: str) -> Dict[str, Any]:
        """Parse AsymptoticLimits results."""
        # Placeholder implementation
        return {
            "observed": 1.0,
            "expected": 1.0,
            "expected_plus_1sigma": 1.5,
            "expected_minus_1sigma": 0.5,
            "expected_plus_2sigma": 2.0,
            "expected_minus_2sigma": 0.25
        }

    def _parse_fit_diagnostics(self, results_file: str) -> Dict[str, Any]:
        """Parse FitDiagnostics results."""
        # Placeholder implementation
        return {
            "best_fit": 1.0,
            "uncertainty": 0.1,
            "pre_fit": {},
            "post_fit": {}
        }

    def _parse_goodness_of_fit(self, results_file: str) -> Dict[str, Any]:
        """Parse GoodnessOfFit results."""
        # Placeholder implementation
        return {
            "p_value": 0.5,
            "chi2": 10.0,
            "ndof": 8
        }

    def _parse_impacts(self, results_file: str) -> Dict[str, Any]:
        """Parse impacts results."""
        # Placeholder implementation
        return {
            "impacts": {
                "lumi": 0.1,
                "btagSF": 0.05,
                "JES": 0.03
            }
        }
