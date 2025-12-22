#!/bin/sh
ulimit -s unlimited
set -e
cd <full_path>/CMSSW_15_0_17/src
export SCRAM_ARCH=el9_amd64_gcc12
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval `scramv1 runtime -sh`

export PYTHONPATH=<full_path>/CMSSW_15_0_17/src/DarkBottomLine/.local:$PYTHONPATH
export PATH="$HOME/.local/bin:$PATH"
cd <full_path>/CMSSW_15_0_17/src/DarkBottomLine
if [ $1 -eq 0 ]; then
  darkbottomline analyze --config configs/2022.yaml --regions-config configs/regions.yaml --input root://cms-xrd-global.cern.ch//store/mc/Run3Summer22NanoAODv12/DYto2L-2Jets_MLL-50_PTLL-40to100_2J_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v1/2560000/30624dd1-ba96-465e-a745-8ff472357277.root --output outputs/hists/regions_data.pkl --max-events 10000
fi
