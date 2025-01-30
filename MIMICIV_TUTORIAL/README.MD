# MIMIC-IV Example

This is an example of how to extract a MEDS dataset from MIMIC-IV. All scripts in this README are assumed to
be run **not** from this directory but from the root directory of this entire repository (e.g., one directory
up from this one).

For this tutorial make sure you are cd'd into the root directory of the repository.

## Extract MIMIC-IV MEDS Data

To get MEDS data, we can either download pre-extracted data and tasks from gcp (Option 1) or extract the data from scratch (Option 2).

### (Option 1) Download pre-extracted data from gcp

Install the [gcloud client](https://cloud.google.com/sdk/docs/install) and then run the following command to download the MEDS data from the gcp bucket:

```console
export ROOT_DIR=??? # set to the directory in which you want to store all data
export MIMICIV_MEDS_DIR=${ROOT_DIR}/meds/ # set to the directory in which you want to store the raw MIMIC-IV data
export OUTPUT_TABULARIZATION_DIR=${ROOT_DIR}/meds_tab/ # set to the output directory for the tabularized data
export OUTPUT_MODEL_DIR=${OUTPUT_TABULARIZATION_DIR}/results/ # set to the base results directory

cd $MIMICIV_MEDS_DIR
gcloud storage cp gs://ehr_standardization_schema/MEDS_Extract_v0.0.7_test.zip meds_extract_0.0.7_data.zip
unzip meds_extract_0.0.7_data.zip
rm meds_extract_0.0.7_data.zip
```

```console
conda create -n meds_tab python=3.12
conda activate meds_tab
pip install "meds-tab==0.0.5"
```

Next we need to get some labels for our tasks. We will use the `long_los` and `icu_mortality` tasks as examples.

### (Option 1) Download pre-extracted labels from gcp:

```console
TASKS=("long_los" "icu_mortality")
TASKS_DIR="$MIMICIV_MEDS_DIR/tasks/" # set to the directory in which you want to store all tasks

mkdir -p "${TASKS_DIR}" # create a directory for the task

for TASK_NAME in "${TASKS[@]}"
do
    gcloud storage cp "gs://ehr_standardization_schema/benchmark_v1/data/labels/${TASK_NAME}.parquet" "${TASKS_DIR}/${TASK_NAME}/0.parquet"
done
```

### (Option 2) Extract MEDS data from scratch

Follow the instructions in the [MEDS_transforms MIMICIV TUTORIAL](https://github.com/mmcdermott/MEDS_transforms/blob/main/MIMIC-IV_Example/README.md)

### (Option 2) Use ACES to extract labels using a task config definition:

We can manually extract the supervised task labels from our meds dataset using [aces](https://github.com/justin13601/ACES/tree/main). First install aces:

```console
conda create -n aces python=3.12
conda activate aces
pip install es-aces==0.5.0
pip install hydra-joblib-launcher
```

Second, run the following command to extract the supervised task labels:

```console
TASKS=(
    "mortality/in_hospital/first_24h"
    "mortality/in_icu/first_24h"
    "mortality/post_hospital_discharge/1y"
    "readmission/30d"
)
TASKS_DIR="$MIMICIV_MEDS_DIR/tasks/" # set to the directory in which you want to store all tasks

for TASK_NAME in "${TASKS[@]}"; do
    SINGLE_TASK_DIR="${MIMICIV_MEDS_DIR}/tasks/${TASK_NAME}"
    mkdir -p $SINGLE_TASK_DIR # create a directory for the task
    cp MIMICIV_TUTORIAL/tasks/${TASK_NAME}.yaml "${SINGLE_TASK_DIR}.yaml"
    aces-cli --multirun hydra/launcher=joblib data=sharded data.standard=meds data.root="$MIMICIV_MEDS_DIR/data" "data.shard=$(expand_shards $MIMICIV_MEDS_DIR/data)" cohort_dir="$TASKS_DIR" cohort_name="$TASK_NAME"
done
```

## Tabularization Workflows

### (Recommended Approach) Selective Task Tabularization

When working with only one or a few tasks of interest, use the selective tabularization script. This approach minimizes computational overhead by only processing event times for specific tasks:

```console
export N_PARALLEL_WORKERS=48 # Set number of workers
TASKS_STR=$(echo ${TASKS[@]} | tr ' ' ',')
bash MIMICIV_TUTORIAL/task_tabularize_meds.sh "${MIMICIV_MEDS_DIR}" $OUTPUT_TABULARIZATION_DIR \
    "${TASKS_STR}" $TASKS_DIR $OUTPUT_MODEL_DIR $N_PARALLEL_WORKERS \
    "tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]" \
    "tabularization.window_sizes=[2h,12h,1d,7d,30d,365d,full]"
```

### (Not Recommended Approach) Full Data Tabularization and XGBoost Baseline

To tabularize all data comprehensively, this script performs a one-time, extensive tabularization of every unique subject_id across all time points. This is an incredibly storage and memory-intensive operation that creates a complete tabular representation of the entire dataset.

The key advantages of this approach include:

- Generates a full tabular dataset once, which can be reused for multiple prediction tasks
- Enables efficient subselection of relevant rows for each different prediction task

We recommend using this script, as opposed to the task specific script, when tabularizing all unique event times would be faster than tabularizing only the relevant events for each task separately.

```console
export N_PARALLEL_WORKERS=48 # Set number of workers
export RESHARD_DIR=${ROOT_DIR}/reshareded_meds/ # set to directory to output reshareded meds data
TASKS_STR=$(echo ${TASKS[@]} | tr ' ' ',')
bash MIMICIV_TUTORIAL/tabularize_meds.sh "${MIMICIV_MEDS_DIR}" "$RESHARD_DIR" $OUTPUT_TABULARIZATION_DIR \
    "${TASKS_STR}" $TASKS_DIR $OUTPUT_MODEL_DIR $N_PARALLEL_WORKERS \
    "tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]" \
    "tabularization.window_sizes=[2h,12h,1d,7d,30d,365d,full]"
```
