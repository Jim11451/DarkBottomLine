from setuptools import setup, find_packages

# Normalize version format for setuptools compatibility
setup(
    name="darkbottomline",
    version="1.0.0b0",  # Changed from "1.0.0-beta" to avoid normalization warning
    description="Modular Coffea-based analysis framework for CMS Run 3 bbMET analysis",
    author="DarkBottomLine Team",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "coffea>=2024.4.0",
        "awkward>=2.0",
        "uproot>=5.0",
        "correctionlib>=2.3.0",
        "dask[complete]",
        "distributed",
        "pyyaml",
        "numpy",
        "pandas",
        "pyarrow",
        "matplotlib",
        "jupyter",
        "hist",
    ],
    entry_points={
        "console_scripts": [
            "darkbottomline=darkbottomline.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
