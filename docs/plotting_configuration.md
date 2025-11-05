# Plotting Configuration Guide

This guide explains how to configure plot exclusions and settings for the DarkBottomLine framework.

## Overview

The plotting system allows you to customize which plots are generated for different regions and categories. This is useful for:

- Excluding irrelevant variables (e.g., `z_mass` and `z_pt` from Top and W CRs)
- Removing plots that don't make sense for certain regions (e.g., jet3 plots from 1b regions)
- Fine-tuning which plots are generated for each analysis category

## Configuration File

The plotting configuration is stored in `configs/plotting.yaml`. You can also specify a custom config file using the `--plot-config` argument.

### Basic Structure

```yaml
# Variables that should NOT use log scale
no_log_scale_vars:
  - n_jets
  - n_bjets
  - n_muons
  # ...

# Region-specific variable exclusions
region_exclusions:
  "Top": ["z_mass", "z_pt"]
  "Wlnu": ["z_mass", "z_pt"]
  # ...
```

## Region Exclusion Patterns

The exclusion system supports three types of pattern matching:

### 1. Exact Region Match

Match the exact region name:

```yaml
region_exclusions:
  "1b:SR": ["jet3_pt", "jet3_eta"]
  "2b:CR_Top_mu": ["some_variable"]
```

### 2. Pattern Match

Match any region containing the pattern:

```yaml
region_exclusions:
  "Top": ["z_mass", "z_pt"]      # Matches all Top CRs
  "Wlnu": ["z_mass", "z_pt"]     # Matches all W CRs
  "Zll": ["w_mass", "w_pt"]      # Matches all Z CRs
```

### 3. Category Pattern

Match all regions in a category:

```yaml
region_exclusions:
  "1b:": ["jet3_pt", "m_jet1jet3"]  # Matches all 1b regions
  "2b:": ["some_variable"]          # Matches all 2b regions
```

## Default Exclusions

The framework automatically applies these default exclusions:

### Signal Regions (SRs)
- **1b SR**: Excludes jet3 variables and all lepton variables
- **2b SR**: Excludes lepton variables (but includes jet3)

### Control Regions (CRs)
- **1b CRs**: Exclude jet3 variables (≤2 jets)
- **2b CRs**: Include jet3 variables (Top CR may have >3 jets)
- **Top CRs**: Exclude `z_mass` and `z_pt`
- **W CRs**: Exclude `z_mass` and `z_pt`
- **Z CRs**: Include `z_mass` and `z_pt` (not excluded)

## Examples

### Example 1: Exclude Additional Variables from 1b SR

```yaml
region_exclusions:
  "1b:SR":
    - jet3_pt
    - jet3_eta
    - m_jet1jet3
    - custom_variable
```

### Example 2: Exclude Variables from All Top CRs

```yaml
region_exclusions:
  "Top":
    - z_mass
    - z_pt
    - w_mass
    - w_pt
```

### Example 3: Exclude Variables from All 1b Regions

```yaml
region_exclusions:
  "1b:":
    - jet3_pt
    - jet3_eta
    - m_jet1jet3
```

### Example 4: Custom Configuration for Specific Analysis

```yaml
no_log_scale_vars:
  - n_jets
  - n_bjets
  - n_muons
  - n_electrons

region_exclusions:
  # Top CRs: exclude Z-related variables
  "Top":
    - z_mass
    - z_pt
    - mll

  # W CRs: exclude Z-related variables
  "Wlnu":
    - z_mass
    - z_pt
    - mll

  # Z CRs: exclude W-related variables
  "Zll":
    - w_mass
    - w_pt
    - mt

  # 1b SR: exclude jet3 and lepton variables (already default)
  "1b:SR":
    - jet3_pt
    - lep1_pt
```

## Usage

### Using Default Configuration

If `configs/plotting.yaml` exists, it will be automatically loaded:

```bash
darkbottomline make-plots --input results.pkl --save-dir outputs
```

### Using Custom Configuration

```bash
darkbottomline make-plots \
  --input results.pkl \
  --save-dir outputs \
  --plot-config my_custom_plotting.yaml
```

### Programmatic Usage

```python
from darkbottomline.plotting import PlotManager
import yaml

# Load configuration
with open("configs/plotting.yaml", "r") as f:
    plot_config = yaml.safe_load(f)

# Initialize plot manager
plot_manager = PlotManager(config=plot_config)

# Create plots
plot_manager.create_all_plots(results, output_dir, ...)
```

## Variable Naming

When specifying exclusions, use the variable names as they appear in the histogram dictionary. Common variable names include:

- Jet variables: `jet1_pt`, `jet2_pt`, `jet3_pt`, `jet1_eta`, `m_jet1jet2`, `m_jet1jet3`
- Lepton variables: `lep1_pt`, `lep2_pt`, `lep1_eta`, `lep2_eta`, `n_muons`, `n_electrons`
- MET variables: `met`, `met_phi`, `recoil`, `delta_phi`
- Mass variables: `z_mass`, `z_pt`, `w_mass`, `w_pt`, `mll`, `mt`
- Multiplicity: `n_jets`, `n_bjets`, `n_muons`, `n_electrons`, `n_leptons`

## Testing

Run the test suite to verify exclusions work correctly:

```bash
pytest tests/test_plot_exclusions.py -v
```

This will verify that:
- 1b SR excludes jet3 and lepton variables
- 2b SR excludes lepton variables but includes jet3
- Top CRs exclude `z_mass` and `z_pt`
- W CRs exclude `z_mass` and `z_pt`
- Z CRs include `z_mass` and `z_pt`
- Custom exclusions work correctly

## Tips

1. **Start with defaults**: The default exclusions are based on StackPlotter logic and should work for most cases.

2. **Use pattern matching**: Instead of listing every region, use patterns like `"Top"` or `"1b:"` to match multiple regions.

3. **Test your configuration**: Use the test suite to verify your exclusions work as expected.

4. **Check variable names**: Make sure variable names in your config match exactly with the histogram keys.

5. **Merge with defaults**: Your custom exclusions are merged with defaults, so you only need to specify what's different.



