#!/usr/bin/env bash

MEDS_DIR=/storage/shared/meds_tabular_ml/ebcl_dataset/processed/final_cohort
OUTPUT_DIR=/storage/shared/meds_tabular_ml/ebcl_dataset/processed/tabularize
N_PARALLEL_WORKERS="2" #"$3"

# echo "Running identify_columns.py: Caching feature names and frequencies."
# POLARS_MAX_THREADS=32 python scripts/identify_columns.py \
#     MEDS_cohort_dir=$MEDS_DIR \
#     tabularized_data_dir=$OUTPUT_DIR \
#     min_code_inclusion_frequency=1 "window_sizes=[1d, 7d, 30d, 365d, full]" do_overwrite=True

# echo "Running tabularize_static.py: tabularizing static data"
# POLARS_MAX_THREADS=32 python scripts/tabularize_static.py \
#     MEDS_cohort_dir=$MEDS_DIR \
#     tabularized_data_dir=$OUTPUT_DIR \
#     min_code_inclusion_frequency=1 "window_sizes=[1d, 7d, 30d, 365d, full]" do_overwrite=True

echo "Running summarize_over_windows.py with $N_PARALLEL_WORKERS workers in parallel"
POLARS_MAX_THREADS=1 python scripts/summarize_over_windows.py \
    --multirun \
    worker="range(1,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    MEDS_cohort_dir=$MEDS_DIR \
    tabularized_data_dir=$OUTPUT_DIR \
    min_code_inclusion_frequency=1 "window_sizes=[1d, 7d, 30d, 365d, full]" do_overwrite=True
