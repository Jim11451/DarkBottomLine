#!/bin/bash

# DarkBottomLine Framework Environment Setup Script
# This script creates a virtual environment and installs all required packages

set -e  # Exit on any error

echo "=========================================="
echo "DarkBottomLine Framework Environment Setup"
echo "=========================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check Python version (require 3.8+)
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "Error: Python 3.8+ is required. Found: $PYTHON_VERSION"
    exit 1
fi

echo "✓ Python $PYTHON_VERSION detected"

# Create virtual environment
VENV_DIR="venv"
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf "$VENV_DIR"
fi

echo "Creating virtual environment..."
python3 -m venv "$VENV_DIR"

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install core scientific packages
echo "Installing core scientific packages..."
pip install numpy scipy matplotlib pandas

# Install physics analysis packages
echo "Installing physics analysis packages..."
pip install awkward uproot correctionlib

# Install Coffea and related packages
echo "Installing Coffea and related packages..."
pip install coffea dask[complete] distributed

# Install machine learning packages
echo "Installing machine learning packages..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn

# Install plotting packages
echo "Installing plotting packages..."
pip install hist plotly

# Install data handling packages
echo "Installing data handling packages..."
pip install pyarrow parquet

# Install development and testing packages
echo "Installing development packages..."
pip install pytest jupyter ipykernel

# Install YAML and configuration packages
echo "Installing configuration packages..."
pip install pyyaml

# Install the DarkBottomLine package in development mode
echo "Installing DarkBottomLine package..."
pip install -e .

# Create output directories
echo "Creating output directories..."
mkdir -p outputs/{hists,plots,combine,dnn,logs}

# Create a test script to verify installation
echo "Creating installation test script..."
cat > test_installation.py << 'EOF'
#!/usr/bin/env python3
"""
Test script to verify DarkBottomLine installation
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            importlib.import_module(f"{package_name}.{module_name}")
        else:
            importlib.import_module(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name}: {e}")
        return False

def main():
    print("Testing DarkBottomLine installation...")
    print("=" * 50)

    # Core Python packages
    core_packages = [
        "numpy", "scipy", "matplotlib", "pandas", "yaml"
    ]

    # Physics packages
    physics_packages = [
        "awkward", "uproot", "correctionlib"
    ]

    # Coffea packages
    coffea_packages = [
        "coffea", "dask", "distributed"
    ]

    # ML packages
    ml_packages = [
        "torch", "sklearn"
    ]

    # Plotting packages
    plotting_packages = [
        "hist", "plotly"
    ]

    # DarkBottomLine modules
    darkbottomline_modules = [
        "objects", "selections", "corrections", "weights",
        "histograms", "processor", "regions", "analyzer",
        "dnn_trainer", "dnn_inference", "plotting",
        "combine_tools", "diagnostics", "cli"
    ]

    all_passed = True

    print("Core packages:")
    for pkg in core_packages:
        if not test_import(pkg):
            all_passed = False

    print("\nPhysics packages:")
    for pkg in physics_packages:
        if not test_import(pkg):
            all_passed = False

    print("\nCoffea packages:")
    for pkg in coffea_packages:
        if not test_import(pkg):
            all_passed = False

    print("\nMachine learning packages:")
    for pkg in ml_packages:
        if not test_import(pkg):
            all_passed = False

    print("\nPlotting packages:")
    for pkg in plotting_packages:
        if not test_import(pkg):
            all_passed = False

    print("\nDarkBottomLine modules:")
    for module in darkbottomline_modules:
        if not test_import(module, "darkbottomline"):
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All packages installed successfully!")
        print("\nYou can now run the DarkBottomLine framework.")
        print("To activate the environment: source venv/bin/activate")
        print("To run analysis: darkbottomline --help")
    else:
        print("✗ Some packages failed to install.")
        print("Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
EOF

# Run the test script
echo "Testing installation..."
python test_installation.py

# Create activation script
echo "Creating activation script..."
cat > activate_env.sh << 'EOF'
#!/bin/bash
# Activate DarkBottomLine environment
source venv/bin/activate
echo "DarkBottomLine environment activated!"
echo "Available commands:"
echo "  darkbottomline --help"
echo "  python test_installation.py"
EOF

chmod +x activate_env.sh

# Create a quick start guide
echo "Creating quick start guide..."
cat > QUICK_START.md << 'EOF'
# DarkBottomLine Framework - Quick Start

## Environment Setup

1. **Activate the environment:**
   ```bash
   source activate_env.sh
   # or manually:
   source venv/bin/activate
   ```

2. **Test the installation:**
   ```bash
   python test_installation.py
   ```

## Running Analysis

1. **Basic analysis:**
   ```bash
   darkbottomline run --year 2023 --input /path/to/nanoaod.root --output results.pkl
   ```

2. **Multi-region analysis:**
   ```bash
   darkbottomline run --year 2023 --input /path/to/nanoaod.root --output results.pkl --regions configs/regions.yaml
   ```

3. **Generate plots:**
   ```bash
   darkbottomline make-plots --year 2023 --region SR --input-hists results.pkl
   ```

4. **Train DNN:**
   ```bash
   darkbottomline train-dnn --config configs/dnn.yaml --signal signal.parquet --background bkg.parquet
   ```

5. **Run complete workflow:**
   ```bash
   ./scripts/run_all.sh
   ```

## Available Commands

- `darkbottomline run` - Run Coffea analysis
- `darkbottomline train-dnn` - Train parametric DNN
- `darkbottomline make-plots` - Generate data/MC plots
- `darkbottomline make-datacard` - Generate Combine datacards
- `darkbottomline run-combine` - Run Higgs Combine fits
- `darkbottomline make-impact` - Generate impact plots
- `darkbottomline make-pulls` - Generate pull plots
- `darkbottomline make-gof` - Generate goodness-of-fit plots

## Configuration Files

- `configs/2023.yaml` - Main analysis configuration
- `configs/regions.yaml` - Region definitions
- `configs/dnn.yaml` - DNN training configuration
- `configs/combine.yaml` - Combine settings

## Output Structure

```
outputs/
├── hists/          # Coffea histograms
├── plots/          # Data/MC plots
├── combine/        # Combine files
├── dnn/            # DNN models
└── logs/           # Log files
```

## Troubleshooting

If you encounter issues:

1. Check that all packages are installed: `python test_installation.py`
2. Verify your input files are valid ROOT files
3. Check the log files in `outputs/logs/`
4. Ensure you have sufficient disk space for outputs

## Support

For issues or questions, check the documentation in the `docs/` directory.
EOF

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To get started:"
echo "1. Activate the environment: source activate_env.sh"
echo "2. Test the installation: python test_installation.py"
echo "3. Run your analysis: darkbottomline --help"
echo ""
echo "Quick start guide: cat QUICK_START.md"
echo ""
echo "Environment created in: $(pwd)/venv"
echo "Output directories created in: $(pwd)/outputs"
echo ""







