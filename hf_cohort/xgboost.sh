#!/usr/bin/env bash

BASE_DIR=/storage/shared/meds_tabular_ml/ebcl_dataset/processed
TAB_DIR=/storage/shared/meds_tabular_ml/ebcl_dataset/processed/tabularize

python -m scripts.xgboost_sweep MEDS_cohort_dir=$BASE_DIR tabularized_data_dir=$TAB_DIR
