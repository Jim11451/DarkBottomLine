#!/bin/bash

# DarkBottomLine Framework - Complete Workflow Script
# This script runs the complete end-to-end analysis workflow

set -e  # Exit on any error

# Configuration
YEAR=${YEAR:-2023}
CONFIG_DIR=${CONFIG_DIR:-configs}
INPUT_DIR=${INPUT_DIR:-inputs}
OUTPUT_DIR=${OUTPUT_DIR:-outputs}
LOG_DIR=${LOG_DIR:-logs}

# Create output directories
mkdir -p ${OUTPUT_DIR}/{hists,plots,combine,models}
mkdir -p ${LOG_DIR}

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${LOG_DIR}/workflow.log
}

# Error handling
handle_error() {
    log "ERROR: $1"
    log "Workflow failed at step: $2"
    exit 1
}

# Step 1: Basic Analysis
log "Step 1: Running basic analysis..."
darkbottomline run \
    --config ${CONFIG_DIR}/${YEAR}.yaml \
    --input ${INPUT_DIR}/data_${YEAR}.root \
    --output ${OUTPUT_DIR}/hists/basic_analysis.coffea \
    --executor futures \
    --workers 4 \
    --max-events 100000 \
    --log-level INFO \
    2>&1 | tee ${LOG_DIR}/basic_analysis.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    handle_error "Basic analysis failed" "Step 1"
fi

log "Basic analysis completed successfully"

# Step 2: Multi-region Analysis
log "Step 2: Running multi-region analysis..."
darkbottomline analyze \
    --config ${CONFIG_DIR}/${YEAR}.yaml \
    --regions-config ${CONFIG_DIR}/regions.yaml \
    --input ${OUTPUT_DIR}/hists/basic_analysis.coffea \
    --output ${OUTPUT_DIR}/hists/region_analysis.coffea \
    --executor futures \
    --workers 4 \
    --log-level INFO \
    2>&1 | tee ${LOG_DIR}/region_analysis.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    handle_error "Multi-region analysis failed" "Step 2"
fi

log "Multi-region analysis completed successfully"

# Step 3: DNN Training
log "Step 3: Training DNN model..."
darkbottomline train-dnn \
    --config ${CONFIG_DIR}/dnn.yaml \
    --signal ${INPUT_DIR}/signals/*.root \
    --background ${INPUT_DIR}/backgrounds/*.root \
    --output ${OUTPUT_DIR}/models/dnn_model.pt \
    --plot-history \
    --log-level INFO \
    2>&1 | tee ${LOG_DIR}/dnn_training.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    handle_error "DNN training failed" "Step 3"
fi

log "DNN training completed successfully"

# Step 4: Create Plots
log "Step 4: Creating validation plots..."
darkbottomline make-plots \
    --input ${OUTPUT_DIR}/hists/region_analysis.coffea \
    --save-dir ${OUTPUT_DIR}/plots \
    --year ${YEAR} \
    --show-data false \
    --log-level INFO \
    2>&1 | tee ${LOG_DIR}/plotting.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    handle_error "Plot creation failed" "Step 4"
fi

log "Plot creation completed successfully"

# Step 5: Generate Datacards
log "Step 5: Generating Combine datacards..."
darkbottomline make-datacard \
    --input ${OUTPUT_DIR}/hists/region_analysis.coffea \
    --output ${OUTPUT_DIR}/combine \
    --region SR \
    --year ${YEAR} \
    --log-level INFO \
    2>&1 | tee ${LOG_DIR}/datacard_generation.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    handle_error "Datacard generation failed" "Step 5"
fi

log "Datacard generation completed successfully"

# Step 6: Run Combine Fits
log "Step 6: Running Combine fits..."

# Fit Diagnostics
log "Running FitDiagnostics..."
darkbottomline run-combine \
    --mode FitDiagnostics \
    --datacard ${OUTPUT_DIR}/combine/datacard.txt \
    --output ${OUTPUT_DIR}/combine \
    --fit-region SR \
    --include-signal false \
    --log-level INFO \
    2>&1 | tee ${LOG_DIR}/fit_diagnostics.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    handle_error "FitDiagnostics failed" "Step 6"
fi

# Asymptotic Limits
log "Running AsymptoticLimits..."
darkbottomline run-combine \
    --mode AsymptoticLimits \
    --datacard ${OUTPUT_DIR}/combine/datacard.txt \
    --output ${OUTPUT_DIR}/combine \
    --log-level INFO \
    2>&1 | tee ${LOG_DIR}/asymptotic_limits.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    handle_error "AsymptoticLimits failed" "Step 6"
fi

# Goodness of Fit
log "Running GoodnessOfFit..."
darkbottomline run-combine \
    --mode GoodnessOfFit \
    --datacard ${OUTPUT_DIR}/combine/datacard.txt \
    --output ${OUTPUT_DIR}/combine \
    --toys 1000 \
    --log-level INFO \
    2>&1 | tee ${LOG_DIR}/goodness_of_fit.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    handle_error "GoodnessOfFit failed" "Step 6"
fi

log "Combine fits completed successfully"

# Step 7: Create Diagnostic Plots
log "Step 7: Creating diagnostic plots..."

# Impact plots
log "Creating impact plots..."
darkbottomline make-impact \
    --input ${OUTPUT_DIR}/combine/fitDiagnostics.root \
    --output ${OUTPUT_DIR}/plots \
    --log-level INFO \
    2>&1 | tee ${LOG_DIR}/impact_plots.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    handle_error "Impact plot creation failed" "Step 7"
fi

# Pull plots
log "Creating pull plots..."
darkbottomline make-pulls \
    --input ${OUTPUT_DIR}/combine/fitDiagnostics.root \
    --output ${OUTPUT_DIR}/plots \
    --log-level INFO \
    2>&1 | tee ${LOG_DIR}/pull_plots.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    handle_error "Pull plot creation failed" "Step 7"
fi

# GOF plots
log "Creating GOF plots..."
darkbottomline make-gof \
    --input ${OUTPUT_DIR}/combine/gof.root \
    --output ${OUTPUT_DIR}/plots \
    --log-level INFO \
    2>&1 | tee ${LOG_DIR}/gof_plots.log

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    handle_error "GOF plot creation failed" "Step 7"
fi

log "Diagnostic plot creation completed successfully"

# Step 8: Generate Summary Report
log "Step 8: Generating summary report..."

# Create summary JSON
cat > ${OUTPUT_DIR}/summary.json << EOF
{
    "workflow": {
        "year": "${YEAR}",
        "timestamp": "$(date -Iseconds)",
        "status": "completed"
    },
    "steps": [
        {
            "name": "basic_analysis",
            "status": "completed",
            "output": "${OUTPUT_DIR}/hists/basic_analysis.coffea"
        },
        {
            "name": "region_analysis",
            "status": "completed",
            "output": "${OUTPUT_DIR}/hists/region_analysis.coffea"
        },
        {
            "name": "dnn_training",
            "status": "completed",
            "output": "${OUTPUT_DIR}/models/dnn_model.pt"
        },
        {
            "name": "plotting",
            "status": "completed",
            "output": "${OUTPUT_DIR}/plots/"
        },
        {
            "name": "datacard_generation",
            "status": "completed",
            "output": "${OUTPUT_DIR}/combine/"
        },
        {
            "name": "combine_fits",
            "status": "completed",
            "output": "${OUTPUT_DIR}/combine/"
        },
        {
            "name": "diagnostic_plots",
            "status": "completed",
            "output": "${OUTPUT_DIR}/plots/"
        }
    ],
    "outputs": {
        "histograms": "${OUTPUT_DIR}/hists/",
        "plots": "${OUTPUT_DIR}/plots/",
        "combine": "${OUTPUT_DIR}/combine/",
        "models": "${OUTPUT_DIR}/models/",
        "logs": "${LOG_DIR}/"
    }
}
EOF

log "Summary report generated: ${OUTPUT_DIR}/summary.json"

# Final summary
log "=========================================="
log "DarkBottomLine Workflow Completed Successfully!"
log "=========================================="
log "Year: ${YEAR}"
log "Output directory: ${OUTPUT_DIR}"
log "Log directory: ${LOG_DIR}"
log ""
log "Outputs:"
log "  - Histograms: ${OUTPUT_DIR}/hists/"
log "  - Plots: ${OUTPUT_DIR}/plots/"
log "  - Combine files: ${OUTPUT_DIR}/combine/"
log "  - DNN model: ${OUTPUT_DIR}/models/"
log "  - Logs: ${LOG_DIR}/"
log ""
log "Summary report: ${OUTPUT_DIR}/summary.json"
log "=========================================="

# Optional: Open results in browser (if on macOS)
if command -v open >/dev/null 2>&1; then
    log "Opening results directory..."
    open ${OUTPUT_DIR}
fi

log "Workflow completed at $(date)"

