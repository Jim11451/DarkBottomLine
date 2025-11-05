# Analysis Structure Documentation

## Region Naming Convention

All regions in the analysis follow a strict naming convention:

**Format**: `{category}:{region_type}_{channel}`

### Examples:
- `1b:SR` - Signal region, 1 b-tag category
- `2b:SR` - Signal region, 2 b-tag category
- `1b:CR_Wlnu_mu` - W+jets control region, 1b category, muon channel
- `1b:CR_Wlnu_el` - W+jets control region, 1b category, electron channel
- `2b:CR_Top_mu` - Top control region, 2b category, muon channel
- `2b:CR_Top_el` - Top control region, 2b category, electron channel
- `1b:CR_Zll_mu` - Z+jets control region, 1b category, muon channel
- `1b:CR_Zll_el` - Z+jets control region, 1b category, electron channel
- `2b:CR_Zll_mu` - Z+jets control region, 2b category, muon channel
- `2b:CR_Zll_el` - Z+jets control region, 2b category, electron channel

## Region Definition Flow

### 1. Configuration (`configs/regions.yaml`)
Regions are defined with full category and channel specification:
```yaml
regions:
  "1b:CR_Zll_mu":
    description: "Z+jets CR (1b definition) muon channel"
    cuts:
      Nmuons: "==2"
      Nelectrons: "==0"
      # ... more cuts
```

### 2. RegionManager (`darkbottomline/regions.py`)
- Loads all regions from `regions.yaml`
- Preserves full region names (e.g., `1b:CR_Zll_mu`)
- `apply_regions()` returns masks with full region names as keys

### 3. DarkBottomLineAnalyzer (`darkbottomline/analyzer.py`)
- Initializes with `RegionManager(regions_config_path)`
- Creates histograms for each region using `self.region_manager.regions.keys()`
- Processes regions using `region_masks` from `apply_regions()`
- Saves results with full region names preserved

### 4. Results Storage
Results are saved with full region names:
```python
{
    "region_histograms": {
        "1b:SR": {...},
        "1b:CR_Zll_mu": {...},
        "1b:CR_Zll_el": {...},
        # ... etc
    }
}
```

## Important Notes

1. **Region names must include category prefix**: All regions must start with `1b:` or `2b:`
2. **Control regions must specify channel**: CRs must have `_mu` or `_el` suffix
3. **No region name transformation**: Region names are preserved exactly as defined in `regions.yaml`
4. **Old data files**: Data files created before the category/channel format update will have old region names and need to be regenerated

## Verification

To verify the region structure is correct:

```python
from darkbottomline.analyzer import DarkBottomLineAnalyzer
import yaml

# Load config
with open('configs/2024.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize analyzer
analyzer = DarkBottomLineAnalyzer(config, 'configs/regions.yaml')

# Check regions
print("Regions:", list(analyzer.region_manager.regions.keys()))
# Should output:
# ['1b:SR', '2b:SR', '1b:CR_Wlnu_mu', '1b:CR_Wlnu_el',
#  '2b:CR_Top_mu', '2b:CR_Top_el', '1b:CR_Zll_mu',
#  '1b:CR_Zll_el', '2b:CR_Zll_mu', '2b:CR_Zll_el']
```

## Troubleshooting

### Issue: Regions in saved file don't have category prefix

**Cause**: Old data file created before regions.yaml update

**Solution**: Re-run the analysis with the updated `regions.yaml`:
```bash
darkbottomline analyze \
  --config configs/2024.yaml \
  --regions-config configs/regions.yaml \
  --input /path/to/input.root \
  --output outputs/hists/regions_data_new.pkl
```

### Issue: Missing channel-separated CRs

**Cause**: Old regions.yaml didn't have channel separation

**Solution**: Ensure `regions.yaml` has all channel-separated CRs (e.g., `1b:CR_Zll_mu` and `1b:CR_Zll_el` instead of just `1b:CR_Zll`)

### Issue: Region names getting simplified

**Check**:
1. Verify `regions.yaml` has full format
2. Check `RegionManager` loads regions correctly
3. Verify `analyzer.region_manager.regions.keys()` has full names
4. Ensure no code is transforming region names before saving



