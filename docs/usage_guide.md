# Core Usage Guide

We provide a set of core CLI scripts to facilitate the tabularization and modeling of MEDS data. These scripts are designed to be run in sequence to transform raw MEDS data into tabularized data and train a model on the tabularized data.

MEDS-Tab tabularizes **task-specific** features: given a task's labels, it computes features only at their prediction times, and every window ends at the prediction time. Stages 1 and 2 prepare the data; stages 3 and 4 tabularize it for one task; stage 5 trains models.

!!! note "Removed: full tabularization and `meds-tab-cache-task`"

    Versions up to 0.2.0 could also tabularize every event time of every subject once (`meds-tab-tabularize-static` and `-time-series` without `input_label_dir`), then select each task's rows with `meds-tab-cache-task`. That workflow is removed: it did not scale to large datasets, and its finite windows ended at the subject's last event before the prediction time rather than at the prediction time. Its documentation is in the [0.2.0 usage guide](https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/0.2.0/docs/usage_guide.md). To migrate, tabularize each task with its own `input_label_dir` and `output_dir`, as below.

## 1. **`MEDS_transform-reshard_to_split`**

This optional command reshards the data. A core challenge in tabularization is the high memory usage and slow compute time. We shard the data into small shards to reduce the memory usage as we can independently tabularize each shard, and we can reduce cpu time by parallelizing the processing of these shards across workers that are independently processing different shards.

```console
MEDS_transform-reshard_to_split \
  --multirun \
  worker="range(0,6)" \
  hydra/launcher=joblib \
  input_dir="$MEDS_DIR" \
  cohort_dir="$MEDS_RESHARD_DIR" \
  'stages=["reshard_to_split"]' \
  stage="reshard_to_split" \
  stage_configs.reshard_to_split.n_subjects_per_shard=2500
```

??? note "Args Description"

    - `--multirun`: This is an optional argument to specify that the command should be run in parallel. We use this here to parallelize the resharing of the data.
    - `hydra/launcher`: This is an optional argument to specify the launcher. When using multirun you should specify the launcher. We use joblib here which enables parallelization on a single machine.
    - `worker`: When using joblib or a hydra slurm launcher, the range of workers must be defined as it specifies the number of parallel workers to spawn. We use 6 workers here.
    - `input_dir`: The directory containing the MEDS data.
    - `cohort_dir`: The directory to store the resharded data.
    - `stages`: The stages to run. We only run the reshard_to_split stage here. MEDS Transform allows for a sequence of stages to be defined and run which is why this is a list.
    - `stage`: The specific stage to run. We run the reshard_to_split stage here. It must be one of the stages in the `stages` kwarg list.
    - `stage_configs.reshard_to_split.n_subjects_per_shard`: The number of subjects per shard. We use 2500 subjects per shard here.

### Input Data Structure

```text
MEDS_DIR/
│
└─── <SPLIT A>
│   │   <SHARD 0>.parquet
│   │   <SHARD 1>.parquet
│   │   ...
│
└─── <SPLIT B>
    │   <SHARD 0>.parquet
    │   <SHARD 1>.parquet
    │   ...
```

### Output Data Structure (New Files)

```text
MEDS_RESHARD_DIR/
│
└─── <SPLIT A>
│   │   <SHARD 0>.parquet
│   │   <SHARD 1>.parquet
│   │   ...
│
└─── <SPLIT B>
    │   <SHARD 0>.parquet
    │   <SHARD 1>.parquet
    │   ...
```

### Complete Directory Structure

!!! abstract "Stage 0 Directory Structure"

    ??? folder "MEDS_DIR"

        ??? folder "SPLIT A"

            - 📄 SHARD 0.parquet

        ??? folder "SPLIT B"

            - 📄 SHARD 0.parquet

    ??? folder "MEDS_RESHARD_DIR"

        ??? folder "SPLIT A"

            - 📄 SHARD 0.parquet
            - 📄 SHARD 1.parquet
            - 📄 ...

        ??? folder "SPLIT B"

            - 📄 SHARD 0.parquet
            - 📄 SHARD 1.parquet
            - 📄 ...

For the rest of the tutorial we will assume that the data has been reshared into the `MEDS_RESHARD_DIR` directory, but this step is optional, and you could instead use the original data directory, `MEDS_DIR`. If you experience high memory issues in later stages, you should try reducing `stage_configs.reshard_to_split.n_subjects_per_shard` to a smaller number.

## 2. **`meds-tab-describe`**

This command computes the frequencies of different code types over the **training subjects only**. Subjects are assigned to splits by the MEDS subject splits file, `metadata/subject_splits.parquet` under the MEDS root directory, and only subjects whose split is `train` are counted, regardless of how the data shards are organized into directories. This file is required: `meds-tab-describe` raises an error if it is missing. Because these frequencies drive feature selection in later stages (e.g., `min_code_inclusion_count`), fitting them on training subjects only keeps tuning and held-out data from influencing which features are kept. The codes are differentiated into the following categories:

- dynamic codes (codes with timestamps)
- dynamic numeric values (codes with timestamps and numerical values)
- static codes (codes without timestamps)
- static numeric values (codes without timestamps but with numerical values)

This script further caches feature names and frequencies in a dataset stored in a `code_metadata.parquet` file within the `OUTPUT_DIR` argument specified as a hydra-style command line argument.

```console
meds-tab-describe \
    "input_dir=${MEDS_RESHARD_DIR}/data" "output_dir=$OUTPUT_DIR"
```

This stage is not parallelized as it runs very quickly.

??? note "Args Description"

    - `input_dir`: The directory containing the MEDS data.
    - `output_dir`: The directory to store the tabularized data.
    - `meds_root_dir` (optional): The MEDS root directory containing `metadata/subject_splits.parquet`. Defaults to the parent of `input_dir`.

!!! warning "Re-run after upgrading"

    Earlier versions fit these frequencies on all splits. With `do_overwrite=False` (the default), an existing `metadata/codes.parquet` from such a run is reused as-is, so re-run `meds-tab-describe` with `do_overwrite=True` (and re-run the downstream stages) to get train-only frequencies.

### Input Data Structure

```text
MEDS_RESHARD_DIR/
│
└─── data
│   └─── <SPLIT A>
│   │   │   <SHARD 0>.parquet
│   │   │   <SHARD 1>.parquet
│   │   │   ...
│   │
│   └─── <SPLIT B>
│       │   <SHARD 0>.parquet
│       │   <SHARD 1>.parquet
│       │   ...
│
└─── metadata
    │   subject_splits.parquet
```

### Output Data Structure (New Files)

```text
OUTPUT_DIR/
│
└─── metadata
    │   codes.parquet
```

### Complete Directory Structure

!!! abstract "Stage 1 Directory Structure"

    ??? folder "MEDS_DIR"

        ??? folder "SPLIT A"

            - 📄 SHARD 0.parquet

        ??? folder "SPLIT B"

            - 📄 SHARD 0.parquet

    ??? folder "MEDS_RESHARD_DIR"

        ??? folder "SPLIT A"

            - 📄 SHARD 0.parquet
            - 📄 SHARD 1.parquet
            - 📄 ...

        ??? folder "SPLIT B"

            - 📄 SHARD 0.parquet
            - 📄 SHARD 1.parquet
            - 📄 ...

    ??? folder "OUTPUT_DIR"

        ??? folder "metadata"

            - 📄 codes.parquet

## 3. **`meds-tab-tabularize-static`**

Filters the codes by frequency and computes the static features of each label of the task: one row per label, in the order of the label files, holding its subject's static features.

```console
meds-tab-tabularize-static \
    "input_dir=${MEDS_RESHARD_DIR}/data" \
    "output_dir=$OUTPUT_DIR" \
    "input_label_dir=${TASKS_DIR}/${TASK}/" \
    tabularization.min_code_inclusion_count=10 \
    tabularization.window_sizes=[1d,30d,365d,full] \
    do_overwrite=False \
    tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
```

This stage is not parallelized as it runs very quickly.

??? note "Args Description"

    - `input_dir`: The directory containing the MEDS data.
    - `output_dir`: The directory to store the tabularized data of this task.
    - `input_label_dir`: The directory containing the task's labels (see "Label file layout" below).
    - `tabularization.min_code_inclusion_count`: The minimum number of times a code must appear.
    - `tabularization.window_sizes`: The window sizes to use for aggregations.
    - `do_overwrite`: Whether to overwrite existing files.
    - `tabularization.aggs`: The aggregation methods to use.

!!! warning "Label file layout"

    `input_label_dir` must mirror the sharding of the MEDS data: the labels of the subjects in `${MEDS_RESHARD_DIR}/data/<SPLIT>/<SHARD>.parquet` go in `${TASKS_DIR}/${TASK}/<SPLIT>/<SHARD>.parquet`, with columns `subject_id`, `prediction_time` and `boolean_value`.

!!! note "Static numeric values (`static/mean`)"

    The examples on this page leave out `static/mean`, the mean of a subject's static (timeless) numeric values of a code. Add it to `tabularization.aggs` only if your data has such values: if it has none, tabularization stops with `No static features found. Remove the aggregation function static/mean`. Many datasets record only categorical demographics as static and give every numeric measurement (height, weight, birth) a time, so the window features cover them instead. That is the case of the MEDS extractions of MIMIC-IV, eICU and NWICU.

!!! tip "Several tasks"

    Use one `output_dir` per task. Run `meds-tab-describe` once and copy its `metadata/` directory into each task's `output_dir` (e.g. `rsync -r "${OUTPUT_DIR}/metadata/" "${OUTPUT_DIR}/${TASK}/metadata"`), as [`MIMICIV_TUTORIAL/task_tabularize_meds.sh`](https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/main/MIMICIV_TUTORIAL/task_tabularize_meds.sh) does.

!!! note "Code Inclusion Parameters"

    In addition to `min_code_inclusion_count` there are several other parameters that can be set in tabularization to restrict the codes that are included:

    - `allowed_codes`: a list of codes to include in the tabularized data
    - `min_code_inclusion_count`: The minimum number of times a code must appear
    - `min_code_inclusion_frequency`: The minimum normalized frequency required
    - `max_included_codes`: The maximum number of codes to include

### Input Data Structure

```text
# Previous structure plus:
TASKS_DIR/
└─── TASK/
    └─── <SPLIT A>
    │   │   <SHARD 0>.parquet  # labels of the subjects of MEDS_RESHARD_DIR/data/<SPLIT A>/<SHARD 0>.parquet
    │   │   <SHARD 1>.parquet
    │   │   ...
    │
    └─── <SPLIT B>
        │   ...
```

### Output Data Structure (New Files)

```text
OUTPUT_DIR/
└─── tabularize/
    └─── <SPLIT A>
    │   │   <SHARD 0>/none/static/present.npz
    │   │   <SHARD 1>/none/static/present.npz
    │   │   ...
    │
    └─── <SPLIT B>
        │   <SHARD 0>/none/static/present.npz
        │   <SHARD 1>/none/static/present.npz
        │   ...
```

### Complete Directory Structure After Static Tabularization

!!! abstract "Stage 3 Directory Structure"

    ??? folder "MEDS_DIR"

        [Previous structure]

    ??? folder "MEDS_RESHARD_DIR"

        [Previous structure]

    ??? folder "TASKS_DIR"

        ??? folder "TASK"

            ??? folder "SPLIT A"

                - 📄 SHARD 0.parquet
                - 📄 SHARD 1.parquet

            ??? folder "SPLIT B"

                [Similar structure to SPLIT A]

    ??? folder "OUTPUT_DIR"

        ??? folder "metadata"

            - 📄 codes.parquet

        ??? folder "tabularize"

            ??? folder "SPLIT A"

                ??? folder "SHARD 0"

                    ??? folder "none/static"

                        - 📄 present.npz

                ??? folder "SHARD 1"

                    ??? folder "none/static"

                        - 📄 present.npz

            ??? folder "SPLIT B"

                [Similar structure to SPLIT A]

## 4. **`meds-tab-tabularize-time-series`**

This stage computes the time-series features of each label: for every window size and aggregation, a summary of the subject's measurements in the window that ends at the label's prediction time. It is the computationally intensive stage, and uses sparse matrices (scipy.sparse), sharding for parallel processing, and Polars for the window computations.

```console
meds-tab-tabularize-time-series \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    "input_dir=${MEDS_RESHARD_DIR}/data" \
    "output_dir=$OUTPUT_DIR" \
    "input_label_dir=${TASKS_DIR}/${TASK}/" \
    tabularization.min_code_inclusion_count=10 \
    tabularization.window_sizes=[1d,30d,365d,full] \
    do_overwrite=False \
    tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
```

!!! info "Windows end at the prediction time"

    For a label at prediction time `t`, a window of size `w` holds the subject's measurements with a time in `(t - w, t]`, and a `full` window holds every measurement up to and including `t`, wherever the subject's last event before `t` falls. For example, with events on Jan 2, 6, 11, 16, 21 and 26, a label on Jan 10 and a `7d` window, the window is (Jan 3, Jan 10\] and holds the Jan 6 measurements only. See [Terminology](terminology.md#where-a-window-ends).

!!! warning "Memory Usage"

    This stage is the most memory intensive stage! This stage should be parallelized to speed up the processing of the data. If you run out of memory, either reduce the workers or reshard your data with `MEDS_transform-reshard_to_split` setting `stage_configs.reshard_to_split.n_subjects_per_shard` to a smaller number.

!!! warning "Code Inclusion Parameters"

    You must use the same code inclusion parameters (which in this example is just `tabularization.min_code_inclusion_count`) as in the previous stage, `meds-tab-tabularize-static`, to ensure that the same codes are included in the tabularized data.

### Input Data Structure

```text
[Previous structure remains the same]
```

### Output Data Structure (New Files)

```text
OUTPUT_DIR/tabularize/
│
└─── <SPLIT A>
│   │   <SHARD 0>/1d/code/count.npz
│   │   <SHARD 0>/1d/value/sum.npz
|   |   ...
|   |   <SHARD 0>/7d/code/count.npz
│   │   <SHARD 0>/7d/value/sum.npz
│   │   ...
|   |   <SHARD 1>/1d/code/count.npz
│   │   <SHARD 1>/1d/value/sum.npz
│   │   ...
│
└─── <SPLIT B>
    │   [Similar structure to SPLIT A]
```

### Complete Directory Structure

!!! abstract "Stage 4 Directory Structure"

    ??? folder "MEDS_DIR"

        [Previous structure]

    ??? folder "MEDS_RESHARD_DIR"

        [Previous structure]

    ??? folder "TASKS_DIR"

        [Previous structure]

    ??? folder "OUTPUT_DIR"

        ??? folder "metadata"

            - 📄 codes.parquet

        ??? folder "tabularize"

            ??? folder "SPLIT A"

                ??? folder "SHARD 0"

                    ??? folder "none/static"

                        - 📄 present.npz

                    ??? folder "1d"

                        ??? folder "code"

                            - 📄 count.npz

                        ??? folder "value"

                            - 📄 sum.npz

                    ??? folder "7d"

                        ??? folder "code"

                            - 📄 count.npz

                        ??? folder "value"

                            - 📄 sum.npz

                ??? folder "SHARD 1"

                    [Similar structure to SHARD 0]

            ??? folder "SPLIT B"

                [Similar structure to SPLIT A]

## 5. **`meds-tab-model`**

Trains a tabular model on the task's tabularized data (read from `${output_dir}/tabularize`) and its labels (`input_label_dir`). The system incorporates extended memory support through sequential shard loading during training and efficient data loading through custom iterators.

### Single Model Training

```console
meds-tab-model \
    model_launcher=xgboost \
    "input_dir=${MEDS_RESHARD_DIR}/data" \
    "output_dir=$OUTPUT_DIR" \
    "input_label_dir=${TASKS_DIR}/${TASK}/" \
    "output_model_dir=${OUTPUT_MODEL_DIR}/${TASK}/" \
    "task_name=$TASK" \
    tabularization.min_code_inclusion_count=10 \
    tabularization.window_sizes=[1d,30d,365d,full] \
    tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
```

### Hyperparameter Optimization

```console
meds-tab-model \
    --multirun \
    model_launcher=xgboost \
    "input_dir=${MEDS_RESHARD_DIR}/data" \
    "output_dir=$OUTPUT_DIR" \
    "input_label_dir=${TASKS_DIR}/${TASK}/" \
    "output_model_dir=${OUTPUT_MODEL_DIR}/${TASK}/" \
    "task_name=$TASK" \
    "hydra.sweeper.n_trials=1000" \
    "hydra.sweeper.n_jobs=${N_PARALLEL_WORKERS}" \
    tabularization.min_code_inclusion_count=10 \
    tabularization.window_sizes=[1d,30d,365d,full] \
    tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
```

??? note "Args Description for Model Stage"

    - `model_launcher`: Choose from `xgboost`, `lightgbm`, `sparsegb`, `knn_classifier`, `logistic_regression`, `random_forest_classifier`, `sgd_classifier`
    - `input_dir`: The directory containing the MEDS data
    - `output_dir`: The directory storing the task's tabularized data
    - `input_label_dir`: The directory containing the task's labels, as given to tabularization
    - `output_model_dir`: Where to save model outputs
    - `hydra.sweeper.n_trials`: Number of trials for hyperparameter optimization
    - `hydra.sweeper.n_jobs`: Number of parallel jobs for optimization
    - `hydra.launcher.n_jobs`: Number of joblib worker processes (all CPUs by default). Each worker is started with `OMP_NUM_THREADS` set to the CPU count divided by this number, which caps the threads of XGBoost and LightGBM. To run trials one at a time on many threads, set it to 1 along with `hydra.sweeper.n_jobs=1`, and set `model_launcher.model.nthread` (XGBoost) or `model_launcher.model.num_threads` (LightGBM).
    - GPU training: `model_launcher.model.device=cuda` (XGBoost, with a CUDA build) or `model_launcher.model.device_type=gpu` (LightGBM; the pip wheels include this OpenCL backend on Linux and Windows). LightGBM's CUDA backend, `device_type=cuda`, needs a source build: `pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON`.
    - `sparsegb`: MEDS-Tab's own gradient-boosted trees for wide, sparse tabularized data, about 10x faster than LightGBM at equal AUROC on the MIMIC-IV MG tasks. It needs the compiled library (a source install with a C++17 compiler and OpenMP) and handles binary tasks. Each tree uses one window per (code, aggregation), drawn from precomputed layouts (`model_launcher.model.layouts`).
    - A swept parameter given a value on the command line is fixed at that value: the hyperparameter optimization command above fixes `tabularization.min_code_inclusion_count` at 10, although every model launcher's sweep space includes it. Leave it off the command line to let the sweep search it.
    - `data_loading_params.cache_matrices=true`: cache each split's matrix, with every tabularized column, under `cache_dir` (default `<output_dir>/.cache`) the first time it is loaded. Later loads, such as the other trials of a sweep, read the cache and apply their own code filter instead of reading and stacking every block again. The data is the same as without it. It costs about one more copy of the task's tabularized data on disk, is rebuilt when the tabularized files or labels change, and is not used with an imputer or a normalizer.

??? note "Code Inclusion Parameters in Modeling"

    In this modeling stage, you can change the code inclusion parameters from previous stages and treat them as tunable hyperparameters. Additional task-specific parameters include:

    - `min_correlation`: Minimum correlation with target required
    - `max_by_correlation`: Maximum number of codes to include based on correlation with target

??? note "Data Preprocessing Options"

    - **Tree-based methods** (e.g., XGBoost, LightGBM):
        - Insensitive to normalization
        - Generally don't benefit from missing value imputation
        - XGBoost handles missing data natively; LightGBM treats absent entries as zeros (see the note below)
    - **Other supported models**:
        - Support sparse matrices
        - May benefit from normalization or imputation

    Available preprocessing options:

    - *Normalization* (maintains sparsity):
        - `standard_scaler`
        - `max_abs_scaler`
    - *Imputation* (converts to dense format):
        - `mean_imputer`
        - `median_imputer`
        - `mode_imputer`

??? note "Observed zeros vs. no measurement: XGBoost and LightGBM differ"

    Tabularized matrices are sparse, and MEDS-Tab stores an observed value of 0 as an explicit entry, distinct
    from an absent entry (no measurement in the window). The two libraries read this differently:

    - **XGBoost** treats absent entries as missing and stored zeros as observed values, so it can tell them
        apart.
    - **LightGBM** treats both as 0 with its default settings. Setting `zero_as_missing=true` treats both as
        missing; neither setting restores the distinction.

    This affects the value aggregations (`value/sum`, `value/sum_sqd`, `value/min`, `value/max`) and
    `static/mean`, where an observed 0 means something other than no measurement. It does not affect
    `code/count`, `value/count` or `static/present`, which are 0 exactly when nothing was observed.

    When `value/count` is among the aggregations, as in the default configurations, LightGBM can recover the
    distinction from it: a code's `value/count` is positive whenever a value, 0 included, was observed in the
    window. On synthetic data where the label depends only on observed-0 vs. absent, LightGBM reached an AUROC
    of 0.64 from the value column alone and 0.78 once the count column was added; XGBoost reached 0.77 from
    the value column alone. Without `value/count`, or for `static/mean` (pair it with `static/present`),
    prefer XGBoost or add an observation indicator if the distinction matters for your task.

### Input/Output Data Structure

```text
[Previous structure remains the same for input]

# New output structure:
OUTPUT_MODEL_DIR/
└─── TASK/YYYY-MM-DD_HH-MM-SS/
    ├── best_trial/
    │   ├── config.log
    │   ├── performance.log
    │   └── xgboost.json
    ├── hydra/
    │   └── optimization_results.yaml
    └── sweep_results/
        └── TRIAL_*/
            ├── config.log
            ├── performance.log
            └── xgboost.json
```

### Complete Directory Structure

!!! abstract "Final Directory Structure"

    ??? folder "MEDS_DIR"

        [Previous structure]

    ??? folder "MEDS_RESHARD_DIR"

        [Previous structure]

    ??? folder "OUTPUT_DIR"

        [Previous structure]

    ??? folder "OUTPUT_MODEL_DIR"

        ??? folder "TASK/YYYY-MM-DD_HH-MM-SS"

            ??? folder "best_trial"

                - 📄 config.log
                - 📄 performance.log
                - 📄 xgboost.json

            ??? folder "hydra"

                - 📄 optimization_results.yaml

            ??? folder "sweep_results"

                ??? folder "TRIAL_1_ID"

                    - 📄 config.log
                    - 📄 performance.log
                    - 📄 xgboost.json

                ??? folder "TRIAL_2_ID"

                    [Similar structure to TRIAL_1_ID]

??? example "Experimental Feature"

    We also support an autogluon based hyperparameter and model search:

    ```console
    meds-tab-autogluon model_launcher=autogluon \
       "input_dir=${MEDS_RESHARD_DIR}/data" \
       "output_dir=$OUTPUT_DIR" \
       "input_label_dir=${TASKS_DIR}/${TASK}/" \
       "output_model_dir=${OUTPUT_MODEL_DIR}/${TASK}/" \
       "task_name=$TASK"
    ```

    Run `meds-tab-autogluon model_launcher=autogluon --help` to see all kwargs. Autogluon requires a lot of memory as it makes all the sparse matrices dense, and is not recommended for large datasets.
