#!/usr/bin/env python3
"""
Script to check if packages from requirements.txt are installed.
If not, installs them locally.

Creator: ptiwari (main creator)

Created based on instructions to:
- Check if packages from requirements.txt are installed
- Install missing packages locally (not in home directory)
- Show summary of installed/missing packages with names and versions
- Suppress pip output during installation
"""

import sys
import re
import subprocess
import argparse
import importlib
import importlib.metadata
from pathlib import Path


def get_package_version(package_name, local_dir=None):
    """Get installed package version, or None if not installed."""
    # Check local directory first if provided
    if local_dir:
        local_dir = Path(local_dir)
        if local_dir.exists():
            local_dir_str = str(local_dir.absolute())
            if local_dir_str not in sys.path:
                sys.path.insert(0, local_dir_str)

            try:
                import_map = {'pyyaml': 'yaml', 'scikit-learn': 'sklearn'}
                module_name = import_map.get(package_name, package_name)
                module = importlib.import_module(module_name)
                if hasattr(module, '__version__'):
                    return module.__version__
                return "installed"
            except ImportError:
                pass

    # Check system installation
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
    """Parse requirements.txt and return list of (package_name, requirement_line, version_constraint)."""
    requirements = []
    with open(requirements_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract base package name (handle extras like dask[complete])
                pkg_match = re.match(r'^([a-zA-Z0-9_-]+(?:\[[^\]]+\])?)', line)
                if pkg_match:
                    package_spec = pkg_match.group(1)
                    # Extract base package name (remove extras)
                    base_package = re.match(r'^([a-zA-Z0-9_-]+)', package_spec).group(1)
                    # Extract version constraint (everything after package spec)
                    version_part = line[len(package_spec):].strip()
                    requirements.append((base_package, line, version_part))
    return requirements


def install_missing_packages(requirements_path, local_dir, missing_lines):
    """Install missing packages locally."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    # Create temp file with only missing packages
    temp_file = local_dir / ".temp_requirements.txt"
    with open(temp_file, 'w') as f:
        for line in missing_lines:
            f.write(line + "\n")

    # Verify Python and pip are working before attempting installation
    try:
        import subprocess as sp_test
        test_result = sp_test.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if test_result.returncode != 0:
            print(f"Error: pip is not working with {sys.executable}")
            print(f"pip error: {test_result.stderr}")
            print(f"\nTry using a different Python installation or fix your Python environment.")
            return False
    except Exception as e:
        print(f"Error: Cannot verify pip installation: {e}")
        print(f"Python executable: {sys.executable}")
        print(f"\nYour Python installation may be corrupted. Try:")
        print(f"  1. Use a virtual environment: python3 -m venv venv")
        print(f"  2. Use conda/miniforge Python")
        print(f"  3. Reinstall Python")
        return False

    # Set environment variables to force installation to target directory only
    import os
    env = os.environ.copy()
    # Set PYTHONUSERBASE to local_dir to prevent any fallback to ~/.local
    env['PYTHONUSERBASE'] = str(local_dir.absolute())
    # Unset PYTHONHOME to prevent interference
    env.pop('PYTHONHOME', None)

    # Install packages (suppress all output)
    # Add --no-user to prevent any user site-packages installation
    # Add --no-cache-dir to avoid cache issues
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--target", str(local_dir.absolute()),
        "--no-user",  # Prevent installation to user site-packages
        "--no-cache-dir",  # Avoid cache issues
        "-r", str(temp_file),
        "--quiet",
        "--disable-pip-version-check",
        "--no-warn-script-location"
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,  # Capture stderr to show if there's an error
        env=env  # Use modified environment
    )

    # Clean up
    if temp_file.exists():
        temp_file.unlink()

    if result.returncode == 0:
        # Add to sys.path
        local_dir_str = str(local_dir.absolute())
        if local_dir_str not in sys.path:
            sys.path.insert(0, local_dir_str)
        return True

    # If installation failed, show error details
    error_msg = result.stderr.decode('utf-8', errors='ignore') if result.stderr else "Unknown error"
    print(f"Installation failed with exit code {result.returncode}")

    # Check for common Python corruption errors
    if "_posixsubprocess" in error_msg or "msvcrt" in error_msg or "ModuleNotFoundError" in error_msg:
        print(f"\nError: Your Python installation appears to be corrupted.")
        print(f"Python executable: {sys.executable}")
        print(f"\nSolutions:")
        print(f"  1. Use a virtual environment:")
        print(f"     python3 -m venv venv")
        print(f"     source venv/bin/activate")
        print(f"     python check_requirements.py --install")
        print(f"  2. Use conda/miniforge:")
        print(f"     conda create -n darkbottomline python=3.9")
        print(f"     conda activate darkbottomline")
        print(f"  3. Reinstall Python or use a different Python installation")

    return False


def main():
    parser = argparse.ArgumentParser(description="Check and install packages from requirements.txt")
    parser.add_argument('--install', action='store_true', help='Install missing packages locally')
    parser.add_argument('--local-dir', default='./.local', help='Local installation directory')
    parser.add_argument('--requirements', default=None, help='Path to requirements.txt')
    args = parser.parse_args()

    # Find requirements.txt
    script_dir = Path(__file__).parent
    requirements_path = Path(args.requirements) if args.requirements else script_dir / "requirements.txt"

    if not requirements_path.exists():
        print(f"Error: requirements.txt not found at {requirements_path}")
        sys.exit(1)

    # Check local directory (only if it exists, don't create it unless installing)
    local_dir = Path(args.local_dir)
    local_dir_for_check = None

    if local_dir.exists():
        local_dir_str = str(local_dir.absolute())
        if local_dir_str not in sys.path:
            sys.path.insert(0, local_dir_str)
        local_dir_for_check = local_dir
    elif args.install:
        # Only create directory if we're installing
        local_dir.mkdir(parents=True, exist_ok=True)
        local_dir_str = str(local_dir.absolute())
        if local_dir_str not in sys.path:
            sys.path.insert(0, local_dir_str)
        local_dir_for_check = local_dir

    # Parse and check requirements
    requirements = parse_requirements(requirements_path)
    missing = []
    installed = []

    for package_name, requirement_line, version_constraint in requirements:
        version = get_package_version(package_name, local_dir_for_check)
        if version:
            installed.append((package_name, version, version_constraint))
        else:
            missing.append((package_name, requirement_line, version_constraint))

    # Show summary
    total = len(requirements)
    installed_count = len(installed)
    missing_count = len(missing)

    print(f"Found {total} packages in requirements.txt")
    print(f"✓ Installed: {installed_count}/{total}")
    print(f"✗ Missing: {missing_count}/{total}")
    print()

    if installed:
        print("Installed packages:")
        for pkg_name, version, version_constraint in installed:
            constraint_info = f" (required: {version_constraint})" if version_constraint else ""
            print(f"  ✓ {pkg_name} (version: {version}){constraint_info}")
        print()

    if missing:
        print("Missing packages:")
        for pkg_name, requirement_line, version_constraint in missing:
            if version_constraint:
                print(f"  ✗ {pkg_name} (will install: {version_constraint})")
            else:
                print(f"  ✗ {pkg_name} (will install: latest)")
        print()

    if not missing:
        print("All packages are installed.")
        return

    if not args.install:
        print(f"To install missing packages, run:")
        print(f"  python {Path(__file__).name} --install")
        return

    print(f"Installing {missing_count} missing packages...")
    missing_lines = [req_line for _, req_line, _ in missing]
    if install_missing_packages(requirements_path, args.local_dir, missing_lines):
        local_packages_path = Path(args.local_dir).absolute()
        print(f"✓ Packages installed to: {local_packages_path}")
        print(f"\nTo use these packages, add to your Python path:")
        print(f"  export PYTHONPATH={local_packages_path}:$PYTHONPATH")
        print(f"\nOr in your Python scripts, add:")
        print(f"  import sys")
        print(f"  sys.path.insert(0, '{local_packages_path}')")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
