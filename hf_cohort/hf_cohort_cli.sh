#!/usr/bin/env bash
# bash hf_cohort/hf_cohort_e2e.sh hf_cohort 80

METHOD=meds

MEDS_DIR=/storage/shared/meds_tabular_ml/ebcl_dataset/processed
OUTPUT_DIR=/storage/shared/meds_tabular_ml/ebcl_dataset/processed/tabularize
ID=$1
N_PARALLEL_WORKERS="$2"
WINDOW_SIZES="tabularization.window_sizes=[1d,7d,30d,365d,full]"
AGGS="tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]"
MIN_CODE_FREQ=10

echo "Running identify_columns.py: Caching feature names and frequencies."
rm -rf $OUTPUT_DIR
meds-tab-describe MEDS_cohort_dir=$MEDS_DIR

echo "Running tabularize_static.py: tabularizing static data"
meds_tab-tabularize-static \
    MEDS_cohort_dir=$MEDS_DIR \
    tabularization.min_code_inclusion_frequency="$MIN_CODE_FREQ" "$WINDOW_SIZES" do_overwrite=False "$AGGS"


POLARS_MAX_THREADS=1
LOG_DIR="logs/$METHOD/$ID-logs"
mkdir -p $LOG_DIR
{ time \
    mprof run --include-children --exit-code --output "$LOG_DIR/mprofile.dat" \
        meds_tab-tabularize-time-series \
            --multirun \
            worker="range(0,$N_PARALLEL_WORKERS)" \
            hydra/launcher=joblib \
            MEDS_cohort_dir=$MEDS_DIR \
            tabularization.min_code_inclusion_frequency="$MIN_CODE_FREQ" do_overwrite=False \
            "$WINDOW_SIZES" "$AGGS" \
    2> $LOG_DIR/cmd.stderr
} 2> $LOG_DIR/timings.txt

cmd_exit_status=${PIPESTATUS[0]}
# Check the exit status of the second command in the pipeline (mprof run ...)
if [ -n "$cmd_exit_status" ] && [ "$cmd_exit_status" -ne 0 ]; then
    echo "build_dataset.sh failed with status $cmd_exit_status."
    echo "Stderr from build_dataset.sh (see $LOG_DIR/cmd.stderr):"
    tail $LOG_DIR/cmd.stderr
    exit "$cmd_exit_status"
fi
mprof plot -o $LOG_DIR/mprofile.png $LOG_DIR/mprofile.dat
mprof peak $LOG_DIR/mprofile.dat > $LOG_DIR/peak_memory_usage.txt


echo "Running task_specific_caching.py: tabularizing static data"
meds_tab-cache-task \
    MEDS_cohort_dir=$MEDS_DIR \
    tabularization.min_code_inclusion_frequency="$MIN_CODE_FREQ" "$WINDOW_SIZES" do_overwrite=False "$AGGS"

echo "Running xgboost: tabularizing static data"
meds_tab-xgboost \
    MEDS_cohort_dir=$MEDS_DIR \
    tabularization.min_code_inclusion_frequency="$MIN_CODE_FREQ" "$WINDOW_SIZES" do_overwrite=False "$AGGS"



