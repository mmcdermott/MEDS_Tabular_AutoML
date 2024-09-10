# MIMIC-IV Example

This is an example of how to extract a MEDS dataset from MIMIC-IV. All scripts in this README are assumed to
be run **not** from this directory but from the root directory of this entire repository (e.g., one directory
up from this one).

## Extract MIMIC-IV MEDS Data

### Download pre-extracted data from gpc

Install the [gcloud client](https://cloud.google.com/sdk/docs/install) and then run the following command to download the MEDS data from the gcp bucket:

```console
export MIMICIV_MEDS_DIR=??? # set to the directory in which you want to store the raw MIMIC-IV data
export OUTPUT_TABULARIZATION_DIR=??? # set to the output directory for the tabularized data
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

### Download pre-extracted labels from gcp:

```console
TASKS=("long_los" "icu_mortality")
TASKS_DIR="$MIMICIV_MEDS_DIR/tasks/" # set to the directory in which you want to store all tasks

mkdir -p "${TASKS_DIR}" # create a directory for the task

for TASK_NAME in "${TASKS[@]}"
do
    gcloud storage cp "gs://ehr_standardization_schema/benchmark_v1/data/labels/${TASK_NAME}.parquet" "${TASKS_DIR}/${TASK_NAME}/0.parquet"
done
```

## Run Tabularization and XGBoost Baseline

```console
export N_PARALLEL_WORKERS=48 # Set number of workers
export RESHARD_DIR=??? # set to directory to output reshareded meds data
bash MIMICIV_TUTORIAL/tabularize_meds.sh "${MIMICIV_MEDS_DIR}" "$RESHARD_DIR" $OUTPUT_TABULARIZATION_DIR \
    "long_los,icu_mortality" $TASKS_DIR $OUTPUT_MODEL_DIR $N_PARALLEL_WORKERS \
    "tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]" \
    "tabularization.window_sizes=[2h,12h,1d,7d,30d,365d,full]"
```
