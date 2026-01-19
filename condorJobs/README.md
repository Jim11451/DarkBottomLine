# Condor Job Submission Guide

This directory contains scripts for submitting DarkBottomLine analysis jobs to Condor.

## Files

- `runanalysis.sh`: Main job execution script (runs on condor nodes)
- `submit.sub`: Template submit file (used by submit_backgrounds.py)
- `submit_backgrounds.py`: **Simple Python script to submit jobs** ⭐

## Quick Start (Recommended)

**Use the Python script - it's simple!**

```bash
# Submit all bkg_*.txt files automatically
python3 condorJobs/submit_backgrounds.py
```

That's it! The script will:
1. Find all `bkg_*.txt` files in `input/` directory
2. For each background file:
   - Count the number of ROOT files in it
   - Submit N condor jobs (one job per file)
   - Each background gets its own condor cluster
3. Each file from a background runs on a separate node

**Example output:**
```
Background: bkg_ttbar.txt
  Files in background: 5
  Will submit: 5 jobs (1 job per file)
  ✓ Submitted cluster: 12345
  ✓ Jobs: 5 (one per file)
  ✓ Each file will run on a separate node

Background: bkg_diboson.txt
  Files in background: 3
  Will submit: 3 jobs (1 job per file)
  ✓ Submitted cluster: 12346
  ✓ Jobs: 3 (one per file)
  ✓ Each file will run on a separate node
```

## Job Structure

```
Server 1 (Cluster 12345):
  Node 1: Job 0 → processes file 1 from bkg_ttbar.txt
  Node 2: Job 1 → processes file 2 from bkg_ttbar.txt
  Node 3: Job 2 → processes file 3 from bkg_ttbar.txt
  Node 4: Job 3 → processes file 4 from bkg_ttbar.txt
  Node 5: Job 4 → processes file 5 from bkg_ttbar.txt

Server 2 (Cluster 12346):
  Node 1: Job 0 → processes file 1 from bkg_diboson.txt
  Node 2: Job 1 → processes file 2 from bkg_diboson.txt
  Node 3: Job 2 → processes file 3 from bkg_diboson.txt
```

## Python Script Options

```bash
# Basic usage
python3 condorJobs/submit_backgrounds.py

# Specify input directory
python3 condorJobs/submit_backgrounds.py --input-dir input

# Custom configuration
python3 condorJobs/submit_backgrounds.py \
    --config configs/2022.yaml \
    --regions-config configs/regions.yaml \
    --executor futures \
    --chunk-size 50000 \
    --workers 4

# Dry run (see what would be submitted)
python3 condorJobs/submit_backgrounds.py --dry-run

# Help
python3 condorJobs/submit_backgrounds.py --help
```

## Setup

1. **Update paths in `submit.sub`**:
   - Change `x509userproxy` to your proxy path (line 3)

2. **Create output directories**:
   ```bash
   mkdir -p condorJobs/output condorJobs/error condorJobs/log
   ```

3. **Initialize voms proxy** (if needed):
   ```bash
   voms-proxy-init --voms cms --valid 192:00
   cp /tmp/x509up_u$(id -u) /afs/cern.ch/user/u/username/private/
   ```

4. **Create background files**:
   - Put your background files in `input/` directory
   - Name them `bkg_*.txt` (e.g., `bkg_ttbar.txt`, `bkg_diboson.txt`)
   - Each file contains one ROOT file path per line

## Background File Format

Each `bkg_*.txt` file should contain ROOT file paths, one per line:

```
# Comments start with #
root://cms-xrd-global.cern.ch//store/mc/.../file1.root
root://cms-xrd-global.cern.ch//store/mc/.../file2.root
root://cms-xrd-global.cern.ch//store/mc/.../file3.root
```

Empty lines and lines starting with `#` are ignored.

## Output Files

Each job produces an output file:
- Format: `outputs/hists/regions_<bkg_name>_<file_index>.pkl`
- Example: `outputs/hists/regions_bkg_ttbar_0.pkl` (first file from bkg_ttbar.txt)

## Monitoring Jobs

```bash
# Check all job status
condor_q

# Check specific cluster
condor_q 12345

# Check job logs
tail -f condorJobs/output/dbl.*.out
tail -f condorJobs/error/dbl.*.err
```

## Advanced: Manual Submission

If you need to submit manually (not recommended):

1. Edit `submit.sub`:
   - Set `DBL_BKG_FILE=bkg_ttbar.txt` (line 32)
   - Set `queue 5` (line 38) where 5 = number of files

2. Submit:
   ```bash
   condor_submit condorJobs/submit.sub
   ```

## Troubleshooting

1. **Jobs held**: Check error logs in `condorJobs/error/` directory
2. **Proxy expired**: Re-run `voms-proxy-init` and copy to AFS
3. **File not found**: Check that input files exist and paths are correct
4. **Memory issues**: Increase `request_memory` in submit.sub or reduce `--chunk-size`
5. **No background files found**: Check that files are in `input/` directory and named `bkg_*.txt`
