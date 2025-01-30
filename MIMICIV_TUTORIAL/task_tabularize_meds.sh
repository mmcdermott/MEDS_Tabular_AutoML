#!/usr/bin/env bash

set -e

# Function to print help message
print_help() {
    echo "Usage: $0 <MIMICIV_MEDS_DIR> <MIMICIV_MEDS_RESHARD_DIR> <OUTPUT_TABULARIZATION_DIR> <TASKS> <TASKS_DIR> <OUTPUT_MODEL_DIR> <N_PARALLEL_WORKERS> [additional arguments]"
    echo
    echo "Arguments:"
    echo "  MIMICIV_MEDS_DIR            Directory containing MIMIC-IV medications data"
    echo "  MIMICIV_MEDS_RESHARD_DIR    Directory for resharded MIMIC-IV medications data"
    echo "  OUTPUT_TABULARIZATION_DIR   Output directory for tabularized data"
    echo "  TASKS                       Comma-separated list of tasks to run (e.g., 'long_los,icu_mortality')"
    echo "  TASKS_DIR                   Directory containing task-specific data"
    echo "  OUTPUT_MODEL_DIR            Output directory for models"
    echo "  N_PARALLEL_WORKERS          Number of parallel workers to use"
    echo
    echo "Additional arguments will be passed to the underlying commands."
}

# Check for help flag
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    print_help
    exit 0
fi

# Check if we have the minimum required number of arguments
if [ "$#" -lt 6 ]; then
    echo "Error: Not enough arguments provided."
    print_help
    exit 1
fi

# Assign arguments to variables
MIMICIV_MEDS_DIR="$1"
OUTPUT_TABULARIZATION_DIR="$2"
TASKS="$3"
TASKS_DIR="$4"
OUTPUT_MODEL_DIR="$5"
N_PARALLEL_WORKERS="$6"

shift 6

# Split the TASKS string into an array
IFS=',' read -ra TASK_ARRAY <<< "$TASKS"

# Print input arguments
echo "Input arguments:"
echo "MIMICIV_MEDS_DIR: $MIMICIV_MEDS_DIR"
echo "OUTPUT_TABULARIZATION_DIR: $OUTPUT_TABULARIZATION_DIR"
echo "TASKS:" "${TASK_ARRAY[@]}"
echo "TASKS_DIR: $TASKS_DIR"
echo "OUTPUT_MODEL_DIR: $OUTPUT_MODEL_DIR"
echo "N_PARALLEL_WORKERS: $N_PARALLEL_WORKERS"
echo "Additional arguments:" "$@"
echo

# describe codes
echo "Describing codes"
meds-tab-describe \
    "input_dir=${MIMICIV_MEDS_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR"

for TASK in "${TASK_ARRAY[@]}"
do
    mkdir -p "${OUTPUT_TABULARIZATION_DIR}/${TASK}"
    rsync -r "${OUTPUT_TABULARIZATION_DIR}/metadata/" "${OUTPUT_TABULARIZATION_DIR}/${TASK}/metadata"

    echo "Tabularizing static data"
    meds-tab-tabularize-static \
        "input_dir=${MIMICIV_MEDS_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR/${TASK}" \
        do_overwrite=False "input_label_dir=${TASKS_DIR}/${TASK}/" "$@"

    meds-tab-tabularize-time-series \
        --multirun \
        worker="range(0,$N_PARALLEL_WORKERS)" \
        hydra/launcher=joblib \
        "input_dir=${MIMICIV_MEDS_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR/${TASK}" \
        do_overwrite=False "input_label_dir=${TASKS_DIR}/${TASK}/" "$@"

    echo "Running xgboost for task: $TASK"
    meds-tab-xgboost \
        --multirun \
        worker="range(0,$N_PARALLEL_WORKERS)" \
        "input_dir=${MIMICIV_MEDS_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR/${TASK}" \
        "output_model_dir=${OUTPUT_MODEL_DIR}/${TASK}/" "task_name=$TASK" do_overwrite=False \
        "hydra.sweeper.n_trials=1000" "hydra.sweeper.n_jobs=${N_PARALLEL_WORKERS}" \
        "input_tabularized_cache_dir=${OUTPUT_TABULARIZATION_DIR}/tabularize/" \
        "input_label_cache_dir=${TASKS_DIR}/${TASK}/" \
        "$@"
done
