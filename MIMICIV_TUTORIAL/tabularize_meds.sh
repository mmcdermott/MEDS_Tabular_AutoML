#!/usr/bin/env bash

set -e

MIMICIV_MEDS_DIR="$1"
MIMICIV_MEDS_RESHARD_DIR="$2"
OUTPUT_TABULARIZATION_DIR="$3"
TASKS="$4"
TASKS_DIR="$5"
OUTPUT_MODEL_DIR="$6"
N_PARALLEL_WORKERS="$7"

shift 7


IFS=',' read -r -a TASK_ARRAY <<< "$TASKS"

MEDS_transform-reshard_to_split \
  --multirun \
  worker="range(0,$N_PARALLEL_WORKERS)" \
  hydra/launcher=joblib \
  input_dir="$MIMICIV_MEDS_DIR" \
  cohort_dir="$MIMICIV_MEDS_RESHARD_DIR" \
  'stages=["reshard_to_split"]' \
  stage="reshard_to_split" \
  stage_configs.reshard_to_split.n_subjects_per_shard=2500 \
  "polling_time=5"

# describe codes
echo "Describing codes"
meds-tab-describe \
    "input_dir=${MIMICIV_MEDS_RESHARD_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR"

echo "Tabularizing static data"
echo meds-tab-tabularize-static \
    "input_dir=${MIMICIV_MEDS_RESHARD_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR" \
    do_overwrite=False "$@"

meds-tab-tabularize-time-series \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    "input_dir=${MIMICIV_MEDS_RESHARD_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR" \
    do_overwrite=False "$@"


for TASK in "${TASK_ARRAY[@]}"
do
    echo "Running task_specific_caching.py"
    meds-tab-cache-task \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    "input_dir=${MIMICIV_MEDS_RESHARD_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR" \
    "input_label_dir=${TASKS_DIR}" "task_name=${TASK}" do_overwrite=False "$@"

  echo "Running xgboost"
  meds-tab-xgboost \
      --multirun \
      worker="range(0,$N_PARALLEL_WORKERS)" \
      "input_dir=${MIMICIV_MEDS_RESHARD_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR" \
      "output_model_dir=${OUTPUT_MODEL_DIR}/${TASK}/" "task_name=$TASK" do_overwrite=False "$@"
done
