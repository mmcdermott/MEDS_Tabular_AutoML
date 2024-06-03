#!/usr/bin/env bash

N_PARALLEL_WORKERS="$1"
shift 1

echo "Tabularizing Static Data"
python scripts/tabularize_static.py "$@"


echo "Tabularizing Time-Series Data"
python scripts/summarize_over_windows.py \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    "$@"
