#!/usr/bin/env bash

METHOD=meds
N_RUNS="1"
OUTPUT_BASE=results
POLARS_MAX_THREADS=32

MEDS_DIR=/storage/shared/meds_tabular_ml/ebcl_dataset/processed
OUTPUT_DIR=/storage/shared/meds_tabular_ml/ebcl_dataset/processed/tabularize
N_PARALLEL_WORKERS="$1"
WINDOW_SIZES="window_sizes=[1d,7d,30d,365d,full]"
AGGS="aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]"
# WINDOW_SIZES="window_sizes=[1d,7d,30d,365d,full]"
# AGGS="aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]"

echo "Running identify_columns.py: Caching feature names and frequencies."
rm -rf $OUTPUT_DIR
POLARS_MAX_THREADS=32 python scripts/identify_columns.py \
    MEDS_cohort_dir=$MEDS_DIR \
    tabularized_data_dir=$OUTPUT_DIR \
    min_code_inclusion_frequency=1 "$WINDOW_SIZES" do_overwrite=False "$AGGS"

echo "Running tabularize_static.py: tabularizing static data"
POLARS_MAX_THREADS=32 python scripts/tabularize_static.py \
    MEDS_cohort_dir=$MEDS_DIR \
    tabularized_data_dir=$OUTPUT_DIR \
    min_code_inclusion_frequency=1 "$WINDOW_SIZES" do_overwrite=False "$AGGS"

POLARS_MAX_THREADS=1
ID=$RANDOM
LOG_DIR="logs/$METHOD/$ID-logs"
mkdir -p $LOG_DIR
{ time \
    mprof run --include-children --exit-code --output "$LOG_DIR/mprofile.dat" \
         python scripts/summarize_over_windows.py \
            --multirun \
            worker="range(0,$N_PARALLEL_WORKERS)" \
            hydra/launcher=joblib \
            MEDS_cohort_dir=$MEDS_DIR \
            tabularized_data_dir=$OUTPUT_DIR \
            min_code_inclusion_frequency=1 do_overwrite=False \
            "$WINDOW_SIZES" "$AGGS" \
    2> $LOG_DIR/cmd.stderr 
} 2> $LOG_DIR/timings.txt

cmd_exit_status=${PIPESTATUS[0]}
# Check the exit status of the second command in the pipeline (mprof run ...)
if [ -n "$cmd_exit_status" ] && [ "$cmd_exit_status" -ne 0 ]; then
    echo "build_dataset.sh failed with status $cmd_exit_status."
    echo "Stderr from build_dataset.sh (see $LOG_DIR/cmd.stderr):"
    tail $LOG_DIR/cmd.stderr
    exit $cmd_exit_status
fi
mprof plot -o $LOG_DIR/mprofile.png $LOG_DIR/mprofile.dat
mprof peak $LOG_DIR/mprofile.dat > $LOG_DIR/peak_memory_usage.txt
