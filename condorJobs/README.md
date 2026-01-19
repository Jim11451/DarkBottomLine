# Condor Job Submission Guide

This directory contains scripts for submitting DarkBottomLine analysis jobs to Condor.

## Files

- `runanalysis.sh`: Main job execution script
- `submit.sub`: Condor submission file

## Setup

1. **Update paths in `submit.sub`**:
   - Change `x509userproxy` to your proxy path
   - Update any other paths as needed

2. **Create output directories**:
   ```bash
   mkdir -p output error log
   ```

3. **Initialize voms proxy** (if needed):
   ```bash
   voms-proxy-init --voms cms --valid 192:00
   cp /tmp/x509up_u$(id -u) /afs/cern.ch/user/u/username/private/
   ```

## Usage Patterns

### Pattern 1: Process Individual Files from a Background (One File Per Node)

This is the recommended pattern for processing backgrounds with multiple files.

**Setup:**
1. Create background files like `input/bkg_ttbar.txt`, `input/bkg_diboson.txt`, etc.
2. Each file contains one ROOT file path per line
3. Each file from the background gets its own condor job/node

**Example for bkg_ttbar.txt:**

1. Count files in the background:
   ```bash
   grep -v '^#' input/bkg_ttbar.txt | grep -v '^$' | wc -l
   # Output: 5 (for example)
   ```

2. Update `submit.sub`:
   ```bash
   environment = "DBL_CONFIG=configs/2022.yaml \
   DBL_REGIONS_CONFIG=configs/regions.yaml \
   DBL_BKG_FILE=bkg_ttbar.txt \
   DBL_EXECUTOR=futures \
   DBL_CHUNK_SIZE=50000 \
   DBL_WORKERS=4"
   
   queue 5  # Number of files in bkg_ttbar.txt
   ```

3. Submit:
   ```bash
   condor_submit submit.sub
   ```

**Result:**
- Job 0: Processes first file from bkg_ttbar.txt → `outputs/hists/regions_bkg_ttbar_0.pkl`
- Job 1: Processes second file from bkg_ttbar.txt → `outputs/hists/regions_bkg_ttbar_1.pkl`
- Job 2: Processes third file from bkg_ttbar.txt → `outputs/hists/regions_bkg_ttbar_2.pkl`
- etc.

### Pattern 2: Process All Files from a Background in One Job

If you want all files from a background processed together:

```bash
environment = "DBL_CONFIG=configs/2022.yaml \
DBL_REGIONS_CONFIG=configs/regions.yaml \
DBL_INPUT=input/bkg_ttbar.txt \
DBL_EXECUTOR=futures \
DBL_CHUNK_SIZE=50000 \
DBL_WORKERS=4"

queue 1
```

**Result:**
- One job processes all files from bkg_ttbar.txt → `outputs/hists/regions_bkg_ttbar.pkl`

### Pattern 3: Process Multiple Backgrounds

To process multiple backgrounds (bkg_ttbar.txt, bkg_diboson.txt, etc.):

**Option A: Submit separately for each background**
```bash
# For bkg_ttbar.txt
environment = "... DBL_BKG_FILE=bkg_ttbar.txt ..."
queue 5  # Number of files in bkg_ttbar.txt

# Then submit again for bkg_diboson.txt
environment = "... DBL_BKG_FILE=bkg_diboson.txt ..."
queue 3  # Number of files in bkg_diboson.txt
```

**Option B: Use a script to submit all backgrounds**

Create a script `submit_all_backgrounds.sh`:
```bash
#!/bin/bash
for bkg in input/bkg_*.txt; do
    bkg_name=$(basename "$bkg" .txt)
    file_count=$(grep -v '^#' "$bkg" | grep -v '^$' | wc -l)
    
    echo "Submitting $bkg_name with $file_count files..."
    
    condor_submit -append "environment = \"DBL_CONFIG=configs/2022.yaml DBL_REGIONS_CONFIG=configs/regions.yaml DBL_BKG_FILE=$(basename $bkg) DBL_EXECUTOR=futures DBL_CHUNK_SIZE=50000 DBL_WORKERS=4\"" \
                 -append "queue $file_count" \
                 submit.sub
done
```

## Environment Variables

- `DBL_CONFIG`: Configuration file (default: configs/2022.yaml)
- `DBL_REGIONS_CONFIG`: Regions config file (default: configs/regions.yaml)
- `DBL_BKG_FILE`: Background file name (e.g., bkg_ttbar.txt) - each file gets separate job
- `DBL_INPUT`: Input file(s) or .txt file - all files processed together
- `DBL_OUTPUT`: Output file path (auto-generated if not set)
- `DBL_EXECUTOR`: Executor type (iterative, futures, dask) (default: futures)
- `DBL_CHUNK_SIZE`: Chunk size for futures/dask (default: 50000)
- `DBL_WORKERS`: Number of workers (default: 4)
- `DBL_MAX_EVENTS`: Maximum events to process (optional)

## Output Files

- Individual file mode (`DBL_BKG_FILE`): `outputs/hists/regions_<bkg_name>_<file_index>.pkl`
- All files mode (`DBL_INPUT` with .txt): `outputs/hists/regions_<bkg_name>.pkl`
- Single file mode: `outputs/hists/regions_<name>_<ProcId>.pkl`

## Monitoring Jobs

```bash
# Check job status
condor_q

# Check specific job
condor_q <ClusterId>.<ProcId>

# Check job logs
tail -f output/dbl.*.out
tail -f error/dbl.*.err
```

## Troubleshooting

1. **Jobs held**: Check error logs in `error/` directory
2. **Proxy expired**: Re-run `voms-proxy-init` and copy to AFS
3. **File not found**: Check that input files exist and paths are correct
4. **Memory issues**: Increase `request_memory` in submit.sub or reduce `DBL_CHUNK_SIZE`
