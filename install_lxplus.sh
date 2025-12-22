#!/bin/bash
# Script to install DarkBottomLine on lxplus (CERN computing environment)
# Creator: ptiwari (main creator)
#
# This script:
# 1. Checks for pre-installed packages in the environment
# 2. Installs missing packages to .local directory
# 3. Installs the DarkBottomLine repository itself

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="${SCRIPT_DIR}/.local"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

echo "=========================================="
echo "DarkBottomLine Installation for lxplus"
echo "=========================================="
echo ""

# Check if requirements.txt exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: requirements.txt not found at $REQUIREMENTS_FILE"
    exit 1
fi

# Step 1: Check for pre-installed packages
echo "Step 1: Checking for pre-installed packages..."
echo "----------------------------------------"

# Use Python to check packages - pass requirements file via environment variable
export REQUIREMENTS_FILE_PATH="$REQUIREMENTS_FILE"
python3 << 'PYTHON_SCRIPT'
import sys
import os
import importlib
import importlib.metadata
import re
from pathlib import Path

def get_package_version(package_name):
    """Get installed package version, or None if not installed."""
    try:
        dist = importlib.metadata.distribution(package_name)
        return dist.version
    except importlib.metadata.PackageNotFoundError:
        try:
            import_map = {'pyyaml': 'yaml', 'scikit-learn': 'sklearn'}
            module_name = import_map.get(package_name, package_name)
            module = importlib.import_module(module_name)
            if hasattr(module, '__version__'):
                return module.__version__
            return "installed"
        except (ImportError, AttributeError):
            return None

def parse_requirements(requirements_path):
    """Parse requirements.txt and return list of (package_name, requirement_line)."""
    requirements = []
    with open(requirements_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                pkg_match = re.match(r'^([a-zA-Z0-9_-]+)', line)
                if pkg_match:
                    requirements.append((pkg_match.group(1), line))
    return requirements

requirements_path = os.environ.get('REQUIREMENTS_FILE_PATH')
if not requirements_path:
    print("Error: REQUIREMENTS_FILE_PATH not set")
    sys.exit(1)

requirements = parse_requirements(requirements_path)

installed = []
missing = []

for package_name, requirement_line in requirements:
    version = get_package_version(package_name)
    if version:
        installed.append((package_name, version))
    else:
        missing.append((package_name, requirement_line))

total = len(requirements)
installed_count = len(installed)
missing_count = len(missing)

print(f"Found {total} packages in requirements.txt")
print(f"✓ Pre-installed: {installed_count}/{total}")
print(f"✗ Missing: {missing_count}/{total}")
print()

if installed:
    print("Pre-installed packages:")
    for pkg_name, version in installed:
        print(f"  ✓ {pkg_name} (version: {version})")
    print()

if missing:
    print("Missing packages:")
    for pkg_name, _ in missing:
        print(f"  ✗ {pkg_name}")
    print()

# Write missing packages to a file for the bash script to use
if missing:
    with open('.missing_packages.txt', 'w') as f:
        for pkg_name, req_line in missing:
            f.write(f"{req_line}\n")
    sys.exit(1)  # Signal that packages are missing
else:
    print("All packages are pre-installed in the environment.")
    sys.exit(0)  # All packages are available
PYTHON_SCRIPT

CHECK_RESULT=$?

if [ $CHECK_RESULT -eq 0 ]; then
    echo "All required packages are already available in the environment."
    echo ""
else
    # Step 2: Install missing packages to .local
    echo "Step 2: Installing missing packages to .local directory..."
    echo "----------------------------------------"

    # Create .local directory if it doesn't exist
    mkdir -p "$LOCAL_DIR"

    # Create temporary requirements file with only missing packages
    TEMP_REQ_FILE="${LOCAL_DIR}/.temp_requirements.txt"

    if [ -f ".missing_packages.txt" ]; then
        cp .missing_packages.txt "$TEMP_REQ_FILE"
        rm -f .missing_packages.txt
    else
        echo "Error: Could not find missing packages list"
        exit 1
    fi

    echo "Installing packages to: $LOCAL_DIR"
    echo ""

    # Install packages to .local (suppress output)
    python3 -m pip install \
        --target "$LOCAL_DIR" \
        -r "$TEMP_REQ_FILE" \
        --quiet \
        --disable-pip-version-check \
        --no-warn-script-location \
        --no-cache-dir \
        2>&1 | grep -v "^$" || true

    # Clean up temp file
    rm -f "$TEMP_REQ_FILE"

    # Install setuptools dependencies to .local to avoid pkg_resources issues
    echo "Installing setuptools dependencies to .local..."
    python3 -m pip install \
        --target "$LOCAL_DIR" \
        setuptools \
        jaraco.text \
        --quiet \
        --disable-pip-version-check \
        --no-cache-dir \
        2>&1 | grep -v "^$" || true

    # Fix NumPy version if needed (CERN's SciPy 1.10.0 requires NumPy <2.0)
    echo "Checking NumPy version compatibility..."
    NUMPY_VERSION=$(python3 -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "")
    if [ -n "$NUMPY_VERSION" ]; then
        NUMPY_MAJOR=$(echo "$NUMPY_VERSION" | cut -d. -f1)
        if [ "$NUMPY_MAJOR" -ge 2 ]; then
            echo "⚠ NumPy 2.x detected, but CERN's SciPy requires NumPy <2.0"
            echo "Downgrading NumPy to <2.0 in .local..."
            python3 -m pip install \
                --target "$LOCAL_DIR" \
                "numpy>=1.21.0,<2.0.0" \
                --force-reinstall \
                --quiet \
                --disable-pip-version-check \
                --no-cache-dir \
                2>&1 | grep -v "^$" || true
        fi
    fi

    echo "✓ Packages installed to .local directory"
    echo ""

    # Add .local to PYTHONPATH
    export PYTHONPATH="${LOCAL_DIR}:${PYTHONPATH}"
    echo "Added .local to PYTHONPATH for this session"
    echo "To make it permanent, add to your ~/.bashrc:"
    echo "  export PYTHONPATH=\"${LOCAL_DIR}:\$PYTHONPATH\""
    echo ""
fi

# Step 3: Install the repository itself
echo "Step 3: Installing DarkBottomLine repository..."
echo "----------------------------------------"

# Temporarily remove .local from PYTHONPATH to avoid conflicts during installation
# We'll add it back after
OLD_PYTHONPATH="$PYTHONPATH"
if [[ "$PYTHONPATH" == *"${LOCAL_DIR}"* ]]; then
    # Remove .local from PYTHONPATH temporarily
    export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^${LOCAL_DIR}$" | tr '\n' ':' | sed 's/:$//')
fi

# Install the package in development mode
echo "Running: pip3 install -e ."
echo ""

if pip3 install -e . --user --no-cache-dir; then
    echo "✓ DarkBottomLine repository installed successfully!"
    echo ""

    # Restore PYTHONPATH
    export PYTHONPATH="$OLD_PYTHONPATH"

    # Add ~/.local/bin to PATH if not already there (for darkbottomline command)
    USER_BIN_DIR="$HOME/.local/bin"
    if [ -d "$USER_BIN_DIR" ] && [[ ":$PATH:" != *":$USER_BIN_DIR:"* ]]; then
        export PATH="$USER_BIN_DIR:$PATH"
        echo "Added ~/.local/bin to PATH for this session"
        echo "To make it permanent, add to your ~/.bashrc:"
        echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo ""
    fi

    # Verify installation
    echo "Verifying installation..."
    if python3 -c "from darkbottomline import DarkBottomLineProcessor; print('✓ Import successful')" 2>/dev/null; then
        echo "✓ Package can be imported"
    else
        echo "⚠ Package installed but import test failed"
        echo "  Make sure .local is in your PYTHONPATH:"
        echo "  export PYTHONPATH=\"${LOCAL_DIR}:\$PYTHONPATH\""
    fi

    # Check if command is available
    if command -v darkbottomline &> /dev/null; then
        echo "✓ darkbottomline command is available"
    else
        echo "⚠ darkbottomline command not found in PATH"
        echo "  Make sure ~/.local/bin is in your PATH:"
        echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo "  Or run: python3 -m darkbottomline.cli --help"
    fi

    echo ""
    echo "=========================================="
    echo "Installation Complete!"
    echo "=========================================="
    echo ""
    echo "To use DarkBottomLine:"
    echo "  1. Make sure .local is in PYTHONPATH:"
    echo "     export PYTHONPATH=\"${LOCAL_DIR}:\$PYTHONPATH\""
    echo ""
    echo "  2. Make sure ~/.local/bin is in PATH:"
    echo "     export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
    echo "  3. Run the framework:"
    echo "     darkbottomline --help"
    echo "     # Or if command not found:"
    echo "     python3 -m darkbottomline.cli --help"
    echo ""
    echo "  4. Or use as a Python module:"
    echo "     python3 -c \"from darkbottomline import DarkBottomLineProcessor\""
    echo ""
else
    # Restore PYTHONPATH
    export PYTHONPATH="$OLD_PYTHONPATH"

    echo "✗ Installation failed"
    echo ""
    echo "If you see pkg_resources errors, try:"
    echo "  pip3 install --target ./.local setuptools jaraco.text"
    echo "  pip3 install -e . --user"
    exit 1
fi