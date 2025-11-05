"""
Unit tests for Combine integration functionality.
"""

import pytest
import tempfile
import yaml
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from darkbottomline.combine_tools import CombineDatacardWriter, CombineRunner


class TestCombineDatacardWriter:
    """Test CombineDatacardWriter class."""

    @pytest.fixture
    def combine_config(self):
        """Create temporary Combine configuration."""
        config = {
            "datacard": {
                "processes": {
                    "signal": {
                        "name": "signal",
                        "rate": 1.0,
                        "is_signal": True
                    },
                    "ttbar": {
                        "name": "ttbar",
                        "rate": 1.0,
                        "is_signal": False
                    },
                    "wjets": {
                        "name": "wjets",
                        "rate": 1.0,
                        "is_signal": False
                    }
                },
                "data": {
                    "name": "data_obs",
                    "rate": 1.0
                },
                "systematics": {
                    "lumi": {
                        "type": "lnN",
                        "processes": ["signal", "ttbar", "wjets"],
                        "value": 1.025
                    },
                    "btagSF": {
                        "type": "shape",
                        "processes": ["signal", "ttbar", "wjets"],
                        "value": 1.0
                    }
                },
                "rate_parameters": {
                    "ttbar_CR_to_SR": {
                        "type": "rateParam",
                        "regions": ["TTbarCR"],
                        "processes": ["ttbar"],
                        "value": 1.0,
                        "uncertainty": 0.3
                    }
                }
            },
            "output": {
                "datacard_file": "datacard.txt",
                "shapes_file": "shapes.root"
            },
            "advanced": {
                "combine_commands": {
                    "text2workspace": "text2workspace.py"
                }
            }
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink()

    def test_datacard_writer_initialization(self, combine_config):
        """Test CombineDatacardWriter initialization."""
        with open(combine_config, 'r') as f:
            config = yaml.safe_load(f)

        writer = CombineDatacardWriter(config)

        assert writer.datacard_config["processes"]["signal"]["name"] == "signal"
        assert writer.datacard_config["systematics"]["lumi"]["type"] == "lnN"
        assert writer.output_config["datacard_file"] == "datacard.txt"

    def test_write_datacard(self, combine_config):
        """Test datacard writing."""
        with open(combine_config, 'r') as f:
            config = yaml.safe_load(f)

        writer = CombineDatacardWriter(config)

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock results
            results = {
                "regions": {
                    "SR": {
                        "n_events": 100,
                        "variables": {}
                    }
                }
            }

            # Write datacard
            datacard_file = writer.write_datacard(results, temp_dir, "SR", "2023")

            # Check that file was created
            assert Path(datacard_file).exists()

            # Check datacard content
            with open(datacard_file, 'r') as f:
                content = f.read()

            assert "imax" in content
            assert "jmax" in content
            assert "kmax" in content
            assert "bin" in content
            assert "observation" in content
            assert "process" in content
            assert "rate" in content
            assert "lumi" in content
            assert "btagSF" in content

    def test_write_shapes(self, combine_config):
        """Test shapes file writing."""
        with open(combine_config, 'r') as f:
            config = yaml.safe_load(f)

        writer = CombineDatacardWriter(config)

        # Create temporary output directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock results
            results = {
                "regions": {
                    "SR": {
                        "n_events": 100,
                        "variables": {}
                    }
                }
            }

            # Write shapes
            shapes_file = writer.write_shapes(results, temp_dir, "SR")

            # Check that file was created
            assert Path(shapes_file).exists()
            assert shapes_file.endswith('.root')

    def test_create_workspace(self, combine_config):
        """Test workspace creation."""
        with open(combine_config, 'r') as f:
            config = yaml.safe_load(f)

        writer = CombineDatacardWriter(config)

        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            datacard_file = Path(temp_dir) / "datacard.txt"

            # Create mock datacard
            with open(datacard_file, 'w') as f:
                f.write("imax 1\njmax 2\nkmax 1\n")

            # Mock subprocess.run to avoid actual text2workspace call
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

                # Create workspace
                workspace_file = writer.create_workspace(str(datacard_file), temp_dir)

                # Check that workspace file path is returned
                assert workspace_file.endswith('.root')
                assert Path(workspace_file).name == "workspace.root"

    def test_generate_datacard_content(self, combine_config):
        """Test datacard content generation."""
        with open(combine_config, 'r') as f:
            config = yaml.safe_load(f)

        writer = CombineDatacardWriter(config)

        # Mock results
        results = {
            "regions": {
                "SR": {
                    "n_events": 100,
                    "variables": {}
                }
            }
        }

        # Generate content
        content = writer._generate_datacard_content(results, "SR", "2023")

        # Check content structure
        lines = content.split('\n')
        assert any('imax' in line for line in lines)
        assert any('jmax' in line for line in lines)
        assert any('kmax' in line for line in lines)
        assert any('bin' in line for line in lines)
        assert any('observation' in line for line in lines)
        assert any('process' in line for line in lines)
        assert any('rate' in line for line in lines)
        assert any('lumi' in line for line in lines)
        assert any('btagSF' in line for line in lines)


class TestCombineRunner:
    """Test CombineRunner class."""

    @pytest.fixture
    def combine_config(self):
        """Create temporary Combine configuration."""
        config = {
            "fit": {
                "strategy": "robustHesse",
                "regions": {
                    "SR_only": ["SR"],
                    "SR_plus_CR": ["SR", "TTbarCR", "ZjetsCR"]
                },
                "signal_hypothesis": {
                    "include_signal": True,
                    "signal_mass": 1000,
                    "signal_cross_section": 1.0
                }
            },
            "output": {
                "fit_results": {
                    "asymptotic_limits": "asymptotic_limits.root",
                    "fit_diagnostics": "fitDiagnostics.root",
                    "goodness_of_fit": "gof.root",
                    "impacts": "impacts.json"
                }
            },
            "advanced": {
                "combine_commands": {
                    "combine": "combine",
                    "combine_tool": "combineTool.py"
                },
                "combine_options": {
                    "asymptotic_limits": ["--run=expected", "--rMin=0", "--rMax=10"],
                    "fit_diagnostics": ["--saveShapes", "--saveNormalizations"],
                    "goodness_of_fit": ["--algo=saturated", "--toys=1000"],
                    "impacts": ["--rMin=0", "--rMax=10"]
                }
            }
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        yield temp_path

        # Cleanup
        Path(temp_path).unlink()

    def test_runner_initialization(self, combine_config):
        """Test CombineRunner initialization."""
        with open(combine_config, 'r') as f:
            config = yaml.safe_load(f)

        runner = CombineRunner(config)

        assert runner.fit_config["strategy"] == "robustHesse"
        assert runner.output_config["fit_results"]["asymptotic_limits"] == "asymptotic_limits.root"
        assert runner.advanced_config["combine_commands"]["combine"] == "combine"

    def test_run_asymptotic_limits(self, combine_config):
        """Test AsymptoticLimits execution."""
        with open(combine_config, 'r') as f:
            config = yaml.safe_load(f)

        runner = CombineRunner(config)

        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            datacard_file = Path(temp_dir) / "datacard.txt"

            # Create mock datacard
            with open(datacard_file, 'w') as f:
                f.write("imax 1\njmax 2\nkmax 1\n")

            # Mock subprocess.run to avoid actual combine call
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

                # Run AsymptoticLimits
                results_file = runner.run_asymptotic_limits(str(datacard_file), temp_dir)

                # Check that results file path is returned
                assert results_file.endswith('.root')
                assert Path(results_file).name == "asymptotic_limits.root"

    def test_run_fit_diagnostics(self, combine_config):
        """Test FitDiagnostics execution."""
        with open(combine_config, 'r') as f:
            config = yaml.safe_load(f)

        runner = CombineRunner(config)

        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            datacard_file = Path(temp_dir) / "datacard.txt"

            # Create mock datacard
            with open(datacard_file, 'w') as f:
                f.write("imax 1\njmax 2\nkmax 1\n")

            # Mock subprocess.run to avoid actual combine call
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

                # Run FitDiagnostics
                results_file = runner.run_fit_diagnostics(
                    str(datacard_file), temp_dir, "SR", False
                )

                # Check that results file path is returned
                assert results_file.endswith('.root')
                assert Path(results_file).name == "fitDiagnostics.root"

    def test_run_goodness_of_fit(self, combine_config):
        """Test GoodnessOfFit execution."""
        with open(combine_config, 'r') as f:
            config = yaml.safe_load(f)

        runner = CombineRunner(config)

        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            datacard_file = Path(temp_dir) / "datacard.txt"

            # Create mock datacard
            with open(datacard_file, 'w') as f:
                f.write("imax 1\njmax 2\nkmax 1\n")

            # Mock subprocess.run to avoid actual combine call
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

                # Run GoodnessOfFit
                results_file = runner.run_goodness_of_fit(
                    str(datacard_file), temp_dir, toys=1000
                )

                # Check that results file path is returned
                assert results_file.endswith('.root')
                assert Path(results_file).name == "gof.root"

    def test_run_impacts(self, combine_config):
        """Test impacts calculation."""
        with open(combine_config, 'r') as f:
            config = yaml.safe_load(f)

        runner = CombineRunner(config)

        # Create temporary files
        with tempfile.TemporaryDirectory() as temp_dir:
            datacard_file = Path(temp_dir) / "datacard.txt"

            # Create mock datacard
            with open(datacard_file, 'w') as f:
                f.write("imax 1\njmax 2\nkmax 1\n")

            # Mock subprocess.run to avoid actual combine call
            with patch('subprocess.run') as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

                # Run impacts
                results_file = runner.run_impacts(str(datacard_file), temp_dir)

                # Check that results file path is returned
                assert results_file.endswith('.json')
                assert Path(results_file).name == "impacts.json"

    def test_parse_results(self, combine_config):
        """Test results parsing."""
        with open(combine_config, 'r') as f:
            config = yaml.safe_load(f)

        runner = CombineRunner(config)

        # Test parsing for different modes
        modes = ["AsymptoticLimits", "FitDiagnostics", "GoodnessOfFit", "Impacts"]

        for mode in modes:
            results = runner.parse_results("dummy_file.root", mode)

            assert isinstance(results, dict)
            assert len(results) > 0

    def test_parse_asymptotic_limits(self, combine_config):
        """Test AsymptoticLimits parsing."""
        with open(combine_config, 'r') as f:
            config = yaml.safe_load(f)

        runner = CombineRunner(config)

        results = runner._parse_asymptotic_limits("dummy_file.root")

        assert "observed" in results
        assert "expected" in results
        assert "expected_plus_1sigma" in results
        assert "expected_minus_1sigma" in results
        assert "expected_plus_2sigma" in results
        assert "expected_minus_2sigma" in results

        # Check that values are reasonable
        assert results["observed"] >= 0
        assert results["expected"] >= 0

    def test_parse_fit_diagnostics(self, combine_config):
        """Test FitDiagnostics parsing."""
        with open(combine_config, 'r') as f:
            config = yaml.safe_load(f)

        runner = CombineRunner(config)

        results = runner._parse_fit_diagnostics("dummy_file.root")

        assert "best_fit" in results
        assert "uncertainty" in results
        assert "pre_fit" in results
        assert "post_fit" in results

        # Check that values are reasonable
        assert results["best_fit"] >= 0
        assert results["uncertainty"] >= 0

    def test_parse_goodness_of_fit(self, combine_config):
        """Test GoodnessOfFit parsing."""
        with open(combine_config, 'r') as f:
            config = yaml.safe_load(f)

        runner = CombineRunner(config)

        results = runner._parse_goodness_of_fit("dummy_file.root")

        assert "p_value" in results
        assert "chi2" in results
        assert "ndof" in results

        # Check that values are reasonable
        assert 0 <= results["p_value"] <= 1
        assert results["chi2"] >= 0
        assert results["ndof"] >= 0

    def test_parse_impacts(self, combine_config):
        """Test impacts parsing."""
        with open(combine_config, 'r') as f:
            config = yaml.safe_load(f)

        runner = CombineRunner(config)

        results = runner._parse_impacts("dummy_file.json")

        assert "impacts" in results
        assert isinstance(results["impacts"], dict)
        assert len(results["impacts"]) > 0

        # Check that impact values are reasonable
        for param, impact in results["impacts"].items():
            assert impact >= 0


class TestCombineIntegration:
    """Integration tests for Combine functionality."""

    def test_full_workflow(self):
        """Test complete Combine workflow."""
        # Create comprehensive Combine config
        config = {
            "datacard": {
                "processes": {
                    "signal": {"name": "signal", "rate": 1.0, "is_signal": True},
                    "ttbar": {"name": "ttbar", "rate": 1.0, "is_signal": False},
                    "wjets": {"name": "wjets", "rate": 1.0, "is_signal": False}
                },
                "data": {"name": "data_obs", "rate": 1.0},
                "systematics": {
                    "lumi": {"type": "lnN", "processes": ["signal", "ttbar", "wjets"], "value": 1.025},
                    "btagSF": {"type": "shape", "processes": ["signal", "ttbar", "wjets"], "value": 1.0}
                }
            },
            "fit": {
                "strategy": "robustHesse",
                "signal_hypothesis": {"include_signal": True, "signal_mass": 1000}
            },
            "output": {
                "datacard_file": "datacard.txt",
                "shapes_file": "shapes.root",
                "workspace_file": "workspace.root",
                "fit_results": {
                    "asymptotic_limits": "asymptotic_limits.root",
                    "fit_diagnostics": "fitDiagnostics.root",
                    "goodness_of_fit": "gof.root",
                    "impacts": "impacts.json"
                }
            },
            "advanced": {
                "combine_commands": {
                    "text2workspace": "text2workspace.py",
                    "combine": "combine",
                    "combine_tool": "combineTool.py"
                },
                "combine_options": {
                    "asymptotic_limits": ["--run=expected", "--rMin=0", "--rMax=10"],
                    "fit_diagnostics": ["--saveShapes", "--saveNormalizations"],
                    "goodness_of_fit": ["--algo=saturated", "--toys=1000"],
                    "impacts": ["--rMin=0", "--rMax=10"]
                }
            }
        }

        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            temp_path = f.name

        try:
            # Initialize components
            writer = CombineDatacardWriter(config)
            runner = CombineRunner(config)

            # Mock results
            results = {
                "regions": {
                    "SR": {
                        "n_events": 100,
                        "variables": {}
                    }
                }
            }

            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write datacard
                datacard_file = writer.write_datacard(results, temp_dir, "SR", "2023")
                assert Path(datacard_file).exists()

                # Write shapes
                shapes_file = writer.write_shapes(results, temp_dir, "SR")
                assert Path(shapes_file).exists()

                # Mock subprocess calls
                with patch('subprocess.run') as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

                    # Create workspace
                    workspace_file = writer.create_workspace(datacard_file, temp_dir)
                    assert workspace_file.endswith('.root')

                    # Run AsymptoticLimits
                    limits_file = runner.run_asymptotic_limits(datacard_file, temp_dir)
                    assert limits_file.endswith('.root')

                    # Run FitDiagnostics
                    diagnostics_file = runner.run_fit_diagnostics(datacard_file, temp_dir, "SR", False)
                    assert diagnostics_file.endswith('.root')

                    # Run GoodnessOfFit
                    gof_file = runner.run_goodness_of_fit(datacard_file, temp_dir, toys=1000)
                    assert gof_file.endswith('.root')

                    # Run impacts
                    impacts_file = runner.run_impacts(datacard_file, temp_dir)
                    assert impacts_file.endswith('.json')

                # Parse results
                limits_results = runner.parse_results(limits_file, "AsymptoticLimits")
                diagnostics_results = runner.parse_results(diagnostics_file, "FitDiagnostics")
                gof_results = runner.parse_results(gof_file, "GoodnessOfFit")
                impacts_results = runner.parse_results(impacts_file, "Impacts")

                # Check that all results are valid
                assert isinstance(limits_results, dict)
                assert isinstance(diagnostics_results, dict)
                assert isinstance(gof_results, dict)
                assert isinstance(impacts_results, dict)

        finally:
            # Cleanup
            Path(temp_path).unlink()







