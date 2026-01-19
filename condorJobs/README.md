# Condor Job Submission Guide

This directory contains scripts for submitting DarkBottomLine analysis jobs to Condor.

## Files

- `runanalysis.sh`: Main job execution script (runs on condor nodes)
- `submit.sub`: Template submit file (used by submit_samples.py)
- `submit_samples.py`: **Simple Python script to submit jobs** ⭐
- `samplefiles/`: Directory containing sample files (*.txt)

## Quick Start (Recommended)

**Use the Python script - it's simple!**

```bash
# Submit all *.txt files in samplefiles/ automatically
python3 condorJobs/submit_samples.py
```

That's it! The script will:
1. Find all `*.txt` files in `condorJobs/samplefiles/` directory
2. For each sample file:
   - Count the number of ROOT files in it
   - Submit N condor jobs (one job per file)
   - Each sample gets its own condor cluster
3. Each file from a sample runs on a separate node

**Example output:**
```
Sample: Zto2Nu-2Jets_PTNuNu-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.txt
  Files in sample: 5
  Will submit: 5 jobs (1 job per file)
  ✓ Submitted cluster: 12345
  ✓ Jobs: 5 (one per file)
  ✓ Each file will run on a separate node

Sample: WtoLNu-2Jets_PTLNu-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8.txt
  Files in sample: 3
  Will submit: 3 jobs (1 job per file)
  ✓ Submitted cluster: 12346
  ✓ Jobs: 3 (one per file)
  ✓ Each file will run on a separate node
```

## Job Structure

```
Server 1 (Cluster 12345):
  Node 1: Job 0 → processes file 1 from Zto2Nu-2Jets_....txt
  Node 2: Job 1 → processes file 2 from Zto2Nu-2Jets_....txt
  Node 3: Job 2 → processes file 3 from Zto2Nu-2Jets_....txt
  Node 4: Job 3 → processes file 4 from Zto2Nu-2Jets_....txt
  Node 5: Job 4 → processes file 5 from Zto2Nu-2Jets_....txt

Server 2 (Cluster 12346):
  Node 1: Job 0 → processes file 1 from WtoLNu-2Jets_....txt
  Node 2: Job 1 → processes file 2 from WtoLNu-2Jets_....txt
  Node 3: Job 2 → processes file 3 from WtoLNu-2Jets_....txt
```

## Python Script Options

```bash
# Basic usage (uses samplefiles/ directory by default)
python3 condorJobs/submit_samples.py

# Specify input directory
python3 condorJobs/submit_samples.py --input-dir samplefiles

# Custom configuration
python3 condorJobs/submit_samples.py \
    --config configs/2022.yaml \
    --regions-config configs/regions.yaml \
    --executor futures \
    --chunk-size 50000 \
    --workers 4

# Dry run (see what would be submitted)
python3 condorJobs/submit_samples.py --dry-run

# Help
python3 condorJobs/submit_samples.py --help
```

## Setup

1. **Update paths in `submit.sub`**:
   - Change `x509userproxy` to your proxy path (line 3)

2. **Create log directories** (automatically created by submit_samples.py):
   ```bash
   mkdir -p condorJobs/logs/output condorJobs/logs/error
   ```
   Note: The script will create these automatically, but you can create them manually if needed.

3. **Initialize voms proxy** (if needed):
   ```bash
   voms-proxy-init --voms cms --valid 192:00
   cp /tmp/x509up_u$(id -u) /afs/cern.ch/user/u/username/private/
   ```

4. **Create sample files**:
   - Put your sample files in `condorJobs/samplefiles/` directory
   - Name them `*.txt` (e.g., `Zto2Nu-2Jets_....txt`, `WtoLNu-2Jets_....txt`)
   - Each file contains one ROOT file path per line

## Sample File Format

Each `*.txt` file in `samplefiles/` should contain ROOT file paths, one per line:

```
# Comments start with #
root://cms-xrd-global.cern.ch//store/mc/.../file1.root
root://cms-xrd-global.cern.ch//store/mc/.../file2.root
root://cms-xrd-global.cern.ch//store/mc/.../file3.root
```

Empty lines and lines starting with `#` are ignored.

## Output Files

Each job produces an output file:
- Format: `outputs/hists/regions_<sample_name>_<file_index>.pkl`
- Example: `outputs/hists/regions_Zto2Nu-2Jets_..._0.pkl` (first file from Zto2Nu-2Jets_....txt)

## Monitoring Jobs

```bash
# Check all job status
condor_q

# Check specific cluster
condor_q 12345

# Check job logs
tail -f condorJobs/logs/output/dbl.*.out
tail -f condorJobs/logs/error/dbl.*.err
```

## Advanced: Manual Submission

If you need to submit manually (not recommended):

1. Edit `submit.sub`:
   - Set `DBL_BKG_FILE=Zto2Nu-2Jets_....txt` (line 32)
   - Set `queue 5` (line 38) where 5 = number of files

2. Submit:
   ```bash
   condor_submit condorJobs/submit.sub
   ```

## Troubleshooting

1. **Jobs held**: Check error logs in `condorJobs/logs/error/` directory
2. **Proxy expired**: Re-run `voms-proxy-init` and copy to AFS
3. **File not found**: Check that input files exist and paths are correct
4. **Memory issues**: Increase `request_memory` in submit.sub or reduce `--chunk-size`
5. **No sample files found**: Check that files are in `condorJobs/samplefiles/` directory and named `*.txt`
