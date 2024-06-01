#!/usr/bin/env bash

MEDS_DIR=/storage/shared/meds_tabular_ml/ebcl_dataset/processed/final_cohort
OUTPUT_DIR=/storage/shared/meds_tabular_ml/ebcl_dataset/processed/tabularize
# N_PARALLEL_WORKERS="$1"
WINDOW_SIZES="window_sizes=[1d]"
AGGS="aggs=[code/count,value/sum]"

python /home/nassim/projects/MEDS_Tabular_AutoML/hf_cohort/aces_task_extraction.py \
    MEDS_cohort_dir=$MEDS_DIR \
    tabularized_data_dir=$OUTPUT_DIR \
    min_code_inclusion_frequency=1 do_overwrite=False \
    "$WINDOW_SIZES" "$AGGS"
