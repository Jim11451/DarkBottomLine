# Validation Notebooks

This directory contains validation notebooks for the DarkBottomLine framework. Each notebook focuses on a specific aspect of validation.

## Notebooks

### 01_plot_exclusions_validation.ipynb
**Plot Exclusions Validation**

Validates that plot exclusions are working correctly for different regions and categories:
- Tests 1b SR exclusions (jet3 + leptons)
- Tests 2b SR exclusions (leptons only)
- Tests Top CR exclusions (z_mass, z_pt)
- Tests W CR exclusions (z_mass, z_pt)
- Tests Z CR inclusions (z_mass, z_pt should be included)
- Tests custom exclusions from config

**Usage:** Run this notebook to verify that plot exclusions match the expected behavior based on region type and category.

---

### 02_region_definitions_validation.ipynb
**Region Definitions Validation**

Validates that region definitions are correctly loaded and parsed:
- Validates region configuration loading
- Checks region cuts are properly defined
- Verifies category and region separation
- Tests RegionManager functionality

**Usage:** Run this notebook to verify that all regions are correctly defined in `configs/regions.yaml` and properly loaded by the RegionManager.

---

### 03_histogram_structure_validation.ipynb
**Histogram Structure Validation**

Validates that histograms are correctly defined and structured:
- Checks histogram definitions
- Validates histogram structure
- Tests histogram filling
- Verifies variable names match exclusion patterns

**Usage:** Run this notebook to verify histogram definitions and ensure variable names match the exclusion patterns used in plotting.

---

### 04_plot_output_structure_validation.ipynb
**Plot Output Structure Validation**

Validates that plots are saved in the correct directory structure:
- Checks plot directory structure
- Validates file naming conventions
- Verifies all formats are generated (PNG, PDF, ROOT, TXT)
- Checks region summary placement

**Usage:** Run this notebook after generating plots to verify:
- Directory structure matches expected format: `outputs/plots/{version}/{format}/{category}/{region}/`
- File naming follows convention: `{category}_{region_dir}_{variable_name}.{format}`
- All formats (PNG, PDF, ROOT, TXT) are generated
- Region summary is saved in the version directory

---

### 05_configuration_validation.ipynb
**Configuration Validation**

Validates that all configuration files are correctly formatted and consistent:
- Validates `regions.yaml`
- Validates `plotting.yaml`
- Checks configuration consistency
- Tests configuration loading

**Usage:** Run this notebook to verify that all configuration files are valid YAML and can be loaded correctly by the framework.

---

### 06_data_mc_comparison_validation.ipynb
**Data/MC Comparison Validation**

Validates data/MC plots and compares yields across regions:
- Loads and inspects results
- Compares yields across regions
- Validates plot generation
- Checks data/MC agreement

**Usage:** Run this notebook after running the analysis to:
- Inspect yields per region
- Verify results structure
- Check which variables would be plotted for each region
- Validate data/MC comparison

---

## Running the Notebooks

### Prerequisites

1. Activate the virtual environment:
```bash
source venv/bin/activate
```

2. Install Jupyter if not already installed:
```bash
pip install jupyter ipykernel
```

3. Start Jupyter:
```bash
jupyter notebook
```

### Order of Execution

While the notebooks can be run independently, the recommended order is:

1. **05_configuration_validation.ipynb** - Validate configs first
2. **02_region_definitions_validation.ipynb** - Verify regions are defined correctly
3. **03_histogram_structure_validation.ipynb** - Check histogram definitions
4. **01_plot_exclusions_validation.ipynb** - Test plot exclusions logic
5. **06_data_mc_comparison_validation.ipynb** - After running analysis, validate results
6. **04_plot_output_structure_validation.ipynb** - After generating plots, validate output structure

### Running All Validations

You can also run all validations using pytest (if converted to test format):

```bash
pytest tests/test_plot_exclusions.py -v
```

## Expected Output

Each notebook should:
- Print validation results with ✓ or ✗ indicators
- Show relevant information about the component being validated
- Provide clear error messages if validation fails
- Include assertions to catch issues early

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure the project root is in the Python path (handled in each notebook)
2. **Missing files**: Some notebooks require results files or plots to exist first
3. **Configuration errors**: Check that YAML files are valid syntax

### Getting Help

If a validation fails:
1. Check the error message for specific details
2. Verify that the component being tested has been properly set up
3. Check that all required files exist in the expected locations
4. Review the relevant configuration files for syntax errors

## Adding New Validations

To add a new validation notebook:

1. Create a new notebook with a descriptive name (e.g., `07_new_validation.ipynb`)
2. Follow the structure of existing notebooks:
   - Markdown cell with title and overview
   - Setup cell with imports and path configuration
   - Validation cells with clear test logic
   - Results display cells
3. Update this README with the new notebook description
4. Ensure the notebook can be run independently



