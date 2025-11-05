#!/bin/bash

# Simple requirements installation script for DarkBottomLine
# This script installs all required packages in the current environment

set -e

echo "Installing DarkBottomLine requirements..."

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: No virtual environment detected."
    echo "It's recommended to use a virtual environment."
    echo "Create one with: python3 -m venv venv && source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install requirements
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

# Install the package in development mode
echo "Installing DarkBottomLine package..."
pip install -e .

echo "Installation complete!"
echo "Test with: python test_installation.py"







