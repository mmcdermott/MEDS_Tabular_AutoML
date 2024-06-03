#!/usr/bin/env bash

BASE_DIR=/storage/shared/meds_tabular_ml/ebcl_dataset/processed_bad_code_cohort
TAB_DIR=$BASE_DIR/tabularize
KEEP_IN_MEMORY=True

python -m scripts.xgboost MEDS_cohort_dir=$BASE_DIR tabularized_data_dir=$TAB_DIR iterator.keep_data_in_memory=$KEEP_IN_MEMORY
