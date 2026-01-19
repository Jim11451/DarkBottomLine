#!/bin/bash
# Condor job script for DarkBottomLine analysis
# Usage: runanalysis.sh <ProcId> [optional arguments]
#
# Environment variables that can be set in submit.sub:
#   DBL_CONFIG: Configuration file (default: configs/2022.yaml)
#   DBL_REGIONS_CONFIG: Regions config file (default: configs/regions.yaml)
#   DBL_OUTPUT: Output file path
#   DBL_EXECUTOR: Executor type (iterative, futures, dask) (default: futures)
#   DBL_CHUNK_SIZE: Chunk size for futures/dask (default: 50000)
#   DBL_WORKERS: Number of workers (default: 4)
#   DBL_MAX_EVENTS: Maximum events to process (optional)

set -e  # Exit on error
set -x  # Debug mode - show commands

# Set unlimited stack size
ulimit -s unlimited

# Get script directory and set up paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DBL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${DBL_DIR}"

echo "=========================================="
echo "DarkBottomLine Condor Job"
echo "=========================================="
echo "Job ID: ${1:-0}"
echo "Working directory: ${DBL_DIR}"
echo "Date: $(date)"
echo ""

# Show directory structure after file transfer (for debugging)
echo "=========================================="
echo "Directory Structure (after file transfer)"
echo "=========================================="
echo "Current directory: $(pwd)"
echo ""
echo "Top-level files and directories:"
ls -la | head -20
echo ""

# Source CMSSW environment if available
if [ -f "/cvmfs/cms.cern.ch/cmsset_default.sh" ]; then
    echo "Sourcing CMSSW environment..."
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    if [ -n "${SCRAM_ARCH}" ]; then
        eval `scramv1 runtime -sh` 2>/dev/null || true
    fi
fi

# Source LCG environment (critical for CERN systems)
if [ -f "/cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh" ]; then
    echo "Sourcing LCG environment..."
    source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc11-opt/setup.sh
else
    echo "⚠ Warning: LCG setup script not found. Continuing anyway..."
fi

# Set up DarkBottomLine environment
echo "Setting up DarkBottomLine environment..."
LOCAL_DIR="${DBL_DIR}/.local"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.9")
SITE_PACKAGES_DIR="${LOCAL_DIR}/lib/python${PYTHON_VERSION}/site-packages"
LOCAL_BIN_DIR="${LOCAL_DIR}/bin"

# Add to PYTHONPATH
if [ -d "${SITE_PACKAGES_DIR}" ]; then
    export PYTHONPATH="${SITE_PACKAGES_DIR}:${PYTHONPATH}"
    echo "✓ Added to PYTHONPATH: ${SITE_PACKAGES_DIR}"
elif [ -d "${LOCAL_DIR}" ]; then
    export PYTHONPATH="${LOCAL_DIR}:${PYTHONPATH}"
    echo "✓ Added to PYTHONPATH: ${LOCAL_DIR}"
else
    echo "⚠ Warning: .local directory not found. Dependencies may not be available."
fi

# Add to PATH
if [ -d "${LOCAL_BIN_DIR}" ]; then
    export PATH="${LOCAL_BIN_DIR}:${PATH}"
    echo "✓ Added to PATH: ${LOCAL_BIN_DIR}"
fi

# Verify Python and required modules
echo ""
echo "Verifying environment..."
if ! command -v python3 &> /dev/null; then
    echo "✗ Error: python3 not found"
    exit 1
fi

python3 --version
echo ""

# Set default values from environment or use defaults
CONFIG="${DBL_CONFIG:-configs/2022.yaml}"
REGIONS_CONFIG="${DBL_REGIONS_CONFIG:-configs/regions.yaml}"
EXECUTOR="${DBL_EXECUTOR:-futures}"
CHUNK_SIZE="${DBL_CHUNK_SIZE:-50000}"
WORKERS="${DBL_WORKERS:-4}"
MAX_EVENTS="${DBL_MAX_EVENTS:-}"

# Handle input: DBL_BKG_FILE mode
# Process individual files from a background file
#    - DBL_BKG_FILE=WtoLNu-2Jets_....txt specifies the sample file
#    - ProcId selects which line/file within that sample (0 = first file, 1 = second, etc.)
#    - Each file from the sample gets its own job/node

PROC_ID="${1:-0}"
BKG_FILE="${DBL_BKG_FILE:-}"

if [ -n "${BKG_FILE}" ]; then
    # Mode 1: Process individual files from a background file
    # BKG_FILE is provided as just the filename (e.g., "WtoLNu-2Jets_....txt")
    # It will be transferred by condor directly to the current working directory

    # Check if file exists (it should be transferred by condor)
    if [ ! -f "${BKG_FILE}" ]; then
        echo "✗ Error: Sample file not found: ${BKG_FILE}"
        echo "  Looking for file in current directory..."
        echo "  Current directory: $(pwd)"
        echo "  Files in current directory:"
        ls -la | head -10
        exit 1
    fi

    # Show that the file was found
    echo "✓ Sample file found: ${BKG_FILE}"
    echo "  File size: $(ls -lh "${BKG_FILE}" | awk '{print $5}')"
    echo "  First few lines of file:"
    head -3 "${BKG_FILE}" | sed 's/^/    /'
    echo ""

    # Extract background name
    BKG_NAME=$(basename "${BKG_FILE}" .txt)

    # Get the specific file from the background file based on ProcId
    # Skip comments and empty lines
    INPUT_LINE=$(grep -v '^#' "${BKG_FILE}" | grep -v '^$' | sed -n "$((PROC_ID + 1))p")

    if [ -z "${INPUT_LINE}" ]; then
        TOTAL_FILES=$(grep -v '^#' "${BKG_FILE}" | grep -v '^$' | wc -l)
        echo "✗ Error: ProcId ${PROC_ID} exceeds number of files in ${BKG_FILE} (${TOTAL_FILES} files)"
        exit 1
    fi

    INPUT="${INPUT_LINE}"
    FILE_INDEX="${PROC_ID}"
    echo "✓ Processing file ${FILE_INDEX} from ${BKG_FILE}"
    echo "  Input file: ${INPUT}"

    # Generate output name: regions_<bkg_name>_<file_index>.pkl
    if [ -z "${DBL_OUTPUT}" ]; then
        OUTPUT="outputs/hists/regions_${BKG_NAME}_${FILE_INDEX}.pkl"
    else
        OUTPUT="${DBL_OUTPUT}"
    fi

else
    # Default fallback - use single default file (for testing)
    INPUT="root://cms-xrd-global.cern.ch//store/mc/Run3Summer22NanoAODv12/DYto2L-2Jets_MLL-50_PTLL-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v1/2560000/30624dd1-ba96-465e-a745-8ff472357277.root"
    BKG_NAME="default_${PROC_ID}"
    OUTPUT="outputs/hists/regions_${BKG_NAME}.pkl"
    echo "⚠ No DBL_BKG_FILE set, using default input file"
fi

# Validate configuration files exist
if [ ! -f "${CONFIG}" ]; then
    echo "✗ Error: Configuration file not found: ${CONFIG}"
    exit 1
fi

if [ ! -f "${REGIONS_CONFIG}" ]; then
    echo "✗ Error: Regions configuration file not found: ${REGIONS_CONFIG}"
    exit 1
fi

# Validate input file exists (should be a ROOT file)
if [ ! -f "${INPUT}" ] && [[ ! "${INPUT}" == root://* ]]; then
    echo "✗ Error: Input file not found: ${INPUT}"
    echo "  Looking for: ${INPUT}"
    exit 1
fi

# Show input information
if [ -n "${BKG_FILE}" ]; then
    TOTAL_FILES=$(grep -v '^#' "${BKG_FILE}" | grep -v '^$' | wc -l)
    echo "✓ Processing file ${FILE_INDEX} of ${TOTAL_FILES} from ${BKG_FILE}"
fi

# Create output directory
OUTPUT_DIR=$(dirname "${OUTPUT}")
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Job Configuration"
echo "=========================================="
echo "ProcId: ${1:-0}"
if [ -n "${BKG_FILE}" ]; then
    echo "Background File: ${BKG_FILE}"
    echo "File Index: ${FILE_INDEX}"
fi
echo "Background Name: ${BKG_NAME}"
echo "Config: ${CONFIG}"
echo "Regions Config: ${REGIONS_CONFIG}"
echo "Input: ${INPUT}"
if [ -n "${BKG_FILE}" ]; then
    TOTAL_FILES=$(grep -v '^#' "${BKG_FILE}" | grep -v '^$' | wc -l)
    echo "  → File ${FILE_INDEX} of ${TOTAL_FILES} from ${BKG_FILE}"
fi
echo "Output: ${OUTPUT}"
echo "Executor: ${EXECUTOR}"
echo "Chunk Size: ${CHUNK_SIZE}"
echo "Workers: ${WORKERS}"
if [ -n "${MAX_EVENTS}" ]; then
    echo "Max Events: ${MAX_EVENTS}"
fi
echo ""

# Build command
CMD="python3 -m darkbottomline.cli analyze"
CMD="${CMD} --config ${CONFIG}"
CMD="${CMD} --regions-config ${REGIONS_CONFIG}"
CMD="${CMD} --input ${INPUT}"
CMD="${CMD} --output ${OUTPUT}"
CMD="${CMD} --executor ${EXECUTOR}"
CMD="${CMD} --workers ${WORKERS}"

# Add chunk-size if executor supports it
if [ "${EXECUTOR}" = "futures" ] || [ "${EXECUTOR}" = "dask" ]; then
    CMD="${CMD} --chunk-size ${CHUNK_SIZE}"
fi

# Add max-events if specified
if [ -n "${MAX_EVENTS}" ]; then
    CMD="${CMD} --max-events ${MAX_EVENTS}"
fi

echo "=========================================="
echo "Running Analysis"
echo "=========================================="
echo "Command: ${CMD}"
echo ""

# Run the analysis
START_TIME=$(date +%s)
if eval "${CMD}"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ""
    echo "=========================================="
    echo "Analysis Completed Successfully!"
    echo "=========================================="
    echo "Output: ${OUTPUT}"
    echo "Duration: ${DURATION} seconds"
    echo "Exit code: 0"
    exit 0
else
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo ""
    echo "=========================================="
    echo "Analysis Failed!"
    echo "=========================================="
    echo "Exit code: ${EXIT_CODE}"
    echo "Duration: ${DURATION} seconds"
    exit ${EXIT_CODE}
fi
