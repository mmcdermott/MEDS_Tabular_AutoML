
OUTPUT_DIR=/data/storage/shared/meds_tabular_ml/ebcl_dataset/processed
PATIENTS_PER_SHARD="2500"
CHUNKSIZE="200_000_000"

rm -rf $OUTPUT_DIR

echo "Running shard_events.py"
POLARS_MAX_THREADS=32 python /home/nassim/projects/MEDS_polars_functions/scripts/extraction/shard_events.py \
    raw_cohort_dir=/data/storage/shared/meds_tabular_ml/ebcl_dataset \
    MEDS_cohort_dir=$OUTPUT_DIR \
    event_conversion_config_fp=/data/storage/shared/meds_tabular_ml/ebcl_dataset/cohort.yaml \
    split_fracs.train=0.6666666666666666 split_fracs.tuning=0.16666666666666666 \
    split_fracs.held_out=0.16666666666666666 row_chunksize=$CHUNKSIZE \
    n_patients_per_shard=$PATIENTS_PER_SHARD hydra.verbose=True

echo "Running split_and_shard_patients.py"
POLARS_MAX_THREADS=32 python /home/nassim/projects/MEDS_polars_functions/scripts/extraction/split_and_shard_patients.py \
    raw_cohort_dir=/data/storage/shared/meds_tabular_ml/ebcl_dataset \
    MEDS_cohort_dir=$OUTPUT_DIR \
    event_conversion_config_fp=/data/storage/shared/meds_tabular_ml/ebcl_dataset/cohort.yaml \
    split_fracs.train=0.6666666666666666 split_fracs.tuning=0.16666666666666666 \
    split_fracs.held_out=0.16666666666666666 row_chunksize=$CHUNKSIZE \
    n_patients_per_shard=$PATIENTS_PER_SHARD hydra.verbose=True

echo "Running convert_to_sharded_events.py"
POLARS_MAX_THREADS=32 python /home/nassim/projects/MEDS_polars_functions/scripts/extraction/convert_to_sharded_events.py \
    raw_cohort_dir=/data/storage/shared/meds_tabular_ml/ebcl_dataset \
    MEDS_cohort_dir=$OUTPUT_DIR \
    event_conversion_config_fp=/data/storage/shared/meds_tabular_ml/ebcl_dataset/cohort.yaml \
    split_fracs.train=0.6666666666666666 split_fracs.tuning=0.16666666666666666 \
    split_fracs.held_out=0.16666666666666666 row_chunksize=$CHUNKSIZE \
    n_patients_per_shard=$PATIENTS_PER_SHARD hydra.verbose=True

echo "Running merge_to_MEDS_cohort.py"
POLARS_MAX_THREADS=32 python /home/nassim/projects/MEDS_polars_functions/scripts/extraction/merge_to_MEDS_cohort.py \
    raw_cohort_dir=/data/storage/shared/meds_tabular_ml/ebcl_dataset \
    MEDS_cohort_dir=$OUTPUT_DIR \
    event_conversion_config_fp=/data/storage/shared/meds_tabular_ml/ebcl_dataset/cohort.yaml \
    split_fracs.train=0.6666666666666666 split_fracs.tuning=0.16666666666666666 \
    split_fracs.held_out=0.16666666666666666 row_chunksize=$CHUNKSIZE \
    n_patients_per_shard=$PATIENTS_PER_SHARD hydra.verbose=True
