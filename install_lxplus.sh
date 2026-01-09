#!/bin/bash
# Script to install DarkBottomLine on lxplus (CERN computing environment)
# Creator: ptiwari (main creator)
#
# This script installs the DarkBottomLine repository itself.
# Dependencies should be installed first using: python3 check_requirements.py --install --local-dir ./.local

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="${SCRIPT_DIR}/.local"

echo "=========================================="
echo "DarkBottomLine Installation for lxplus"
echo "=========================================="
echo ""
echo "Note: Make sure dependencies are installed first:"
echo "  python3 check_requirements.py --install --local-dir ./.local"
echo ""

# Add .local to PYTHONPATH if it exists (packages installed by check_requirements.py)
if [ -d "$LOCAL_DIR" ]; then
    export PYTHONPATH="${LOCAL_DIR}:${PYTHONPATH}"
    echo "✓ Found .local directory, added to PYTHONPATH"
    echo ""
fi

# Install the package in development mode
echo "Installing DarkBottomLine repository..."
echo "----------------------------------------"
echo "Running: pip3 install -e ."
echo ""

# Temporarily remove .local from PYTHONPATH to avoid conflicts during installation
# We'll add it back after
OLD_PYTHONPATH="$PYTHONPATH"
if [[ "$PYTHONPATH" == *"${LOCAL_DIR}"* ]]; then
    # Remove .local from PYTHONPATH temporarily
    export PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "^${LOCAL_DIR}$" | tr '\n' ':' | sed 's/:$//')
fi

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
