# Core Usage Guide

We provide a set of core CLI scripts to facilitate the tabularization and modeling of MEDS data. These scripts are designed to be run in sequence to transform raw MEDS data into tabularized data and train a model on the tabularized data.

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

This command processes MEDS data shards to compute the frequencies of different code types. It differentiates codes into the following categories:

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

### Input Data Structure

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

Filters and processes the dataset based on the count of codes, generating a tabular vector for each patient at each timestamp in the shards. Each row corresponds to a unique `subject_id` and `timestamp` combination. As a result, rows are duplicated across multiple timestamps for the same patient.

```console
meds-tab-tabularize-static \
    "input_dir=${MEDS_RESHARD_DIR}/data" \
    "output_dir=$OUTPUT_DIR" \
    tabularization.min_code_inclusion_count=10 \
    tabularization.window_sizes=[1d,30d,365d,full] \
    do_overwrite=False \
    tabularization.aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
```

This stage is not parallelized as it runs very quickly.

??? note "Args Description"
    - `input_dir`: The directory containing the MEDS data.
    - `output_dir`: The directory to store the tabularized data.
    - `tabularization.min_code_inclusion_count`: The minimum number of times a code must appear.
    - `tabularization.window_sizes`: The window sizes to use for aggregations.
    - `do_overwrite`: Whether to overwrite existing files.
    - `tabularization.aggs`: The aggregation methods to use.

!!! note "Code Inclusion Parameters"
    In addition to `min_code_inclusion_count` there are several other parameters that can be set in tabularization to restrict the codes that are included:

    - `allowed_codes`: a list of codes to include in the tabularized data
    - `min_code_inclusion_count`: The minimum number of times a code must appear
    - `min_code_inclusion_frequency`: The minimum normalized frequency required
    - `max_included_codes`: The maximum number of codes to include

### Input Data Structure

```text
[Previous structure remains the same]
```

### Output Data Structure (New Files)

```text
OUTPUT_DIR/
└─── tabularize/
    └─── <SPLIT A>
    │   │   <SHARD 0>/none/static/present.npz
    │   │   <SHARD 0>/none/static/first.npz
    │   │   <SHARD 1>/none/static/present.npz
    │   │   ...
    │
    └─── <SPLIT B>
        │   <SHARD 0>/none/static/present.npz
        │   <SHARD 0>/none/static/first.npz
        │   <SHARD 1>/none/static/present.npz
        │   ...
```

### Complete Directory Structure After Static Tabularization

!!! abstract "Stage 3 Directory Structure"
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

        ??? folder "tabularize"
            ??? folder "SPLIT A"
                ??? folder "SHARD 0"
                    ??? folder "none/static"
                        - 📄 present.npz
                        - 📄 first.npz

                ??? folder "SHARD 1"
                    ??? folder "none/static"
                        - 📄 present.npz
                        - 📄 first.npz

            ??? folder "SPLIT B"
                \[Similar structure to SPLIT A\]

## 4. **`meds-tab-tabularize-time-series`**

This stage handles the computationally intensive task of converting temporal medical data into feature vectors. The process employs several key optimizations: sparse matrix operations utilizing scipy.sparse for memory-efficient storage, data sharding that enables parallel processing, and efficient aggregation using Polars for fast rolling window computations.

```console
meds-tab-tabularize-time-series \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    "input_dir=${MEDS_RESHARD_DIR}/data" \
    "output_dir=$OUTPUT_DIR" \
    tabularization.min_code_inclusion_count=10 \
    tabularization.window_sizes=[1d,30d,365d,full] \
    tabularization.aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
```

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
        \[Previous structure\]

    ??? folder "MEDS_RESHARD_DIR"
        \[Previous structure\]

    ??? folder "OUTPUT_DIR"
        ??? folder "metadata"
            - 📄 codes.parquet

        ??? folder "tabularize"
            ??? folder "SPLIT A"
                ??? folder "SHARD 0"
                    ??? folder "none/static"
                        - 📄 present.npz
                        - 📄 first.npz

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
                    \[Similar structure to SHARD 0\]

            ??? folder "SPLIT B"
                \[Similar structure to SPLIT A\]

## 5. **`meds-tab-cache-task`**

Aligns task-specific labels with the nearest prior event in the tabularized data. It requires a labeled dataset directory with three columns (`subject_id`, `timestamp`, `label`) structured similarly to the `input_dir`.

```console
meds-tab-cache-task \
    --multirun \
    hydra/launcher=joblib \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    "input_dir=${MEDS_RESHARD_DIR}/data" \
    "output_dir=$OUTPUT_DIR" \
    "input_label_dir=${TASKS_DIR}/${TASK}/" \
    "task_name=${TASK}" \
    tabularization.min_code_inclusion_count=10 \
    tabularization.window_sizes=[1d,30d,365d,full] \
    tabularization.aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
```

!!! warning "Stage Duration"
    This stage is the slowest stage, but should not be as memory intensive, so make sure to parallelize across as many workers as possible.

!!! warning "Code Inclusion Parameters"
    You must use the same code inclusion parameters (which in this example is just `tabularization.min_code_inclusion_count`) as in the previous stages to ensure that the same codes are included in the tabularized data.

### Input Data Structure

```text
# Previous structure plus:
TASKS_DIR/
└─── TASK/
    │   *.parquet  # All parquet files containing labels
```

### Output Data Structure (New Files)

```text
OUTPUT_DIR/
└─── TASK/
    ├─── labels/
    │   └─── <SPLIT A>
    │       │   <SHARD 0>.parquet
    │       │   <SHARD 1>.parquet
    │       └─── <SPLIT B>
    │           │   <SHARD 0>.parquet
    │           │   <SHARD 1>.parquet
    └─── task_cache/
        [Similar structure to tabularize/ but filtered for task]
```

### Complete Directory Structure

!!! abstract "Stage 5 Directory Structure"
    ??? folder "MEDS_DIR"
        \[Previous structure\]

    ??? folder "MEDS_RESHARD_DIR"
        \[Previous structure\]

    ??? folder "OUTPUT_DIR"
        ??? folder "metadata"
            - 📄 codes.parquet

        ??? folder "tabularize"
            \[Previous structure\]

        ??? folder "${TASK}"
            ??? folder "labels"
                ??? folder "SPLIT A"
                    - 📄 SHARD 0.parquet
                    - 📄 SHARD 1.parquet

                ??? folder "SPLIT B"
                    - 📄 SHARD 0.parquet
                    - 📄 SHARD 1.parquet

            ??? folder "task_cache"
                ??? folder "SPLIT A"
                    ??? folder "SHARD 0"
                        ??? folder "none/static"
                            - 📄 present.npz
                            - 📄 first.npz

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
                        \[Similar structure to SHARD 0\]

                ??? folder "SPLIT B"
                    \[Similar structure to SPLIT A\]

## 6. **`meds-tab-model`**

Trains a tabular model using user-specified parameters. The system incorporates extended memory support through sequential shard loading during training and efficient data loading through custom iterators.

### Single Model Training

```console
meds-tab-model \
    model_launcher=xgboost \
    "input_dir=${MEDS_RESHARD_DIR}/data" \
    "output_dir=$OUTPUT_DIR" \
    "output_model_dir=${OUTPUT_MODEL_DIR}/${TASK}/" \
    "task_name=$TASK" \
    tabularization.min_code_inclusion_count=10 \
    tabularization.window_sizes=[1d,30d,365d,full] \
    tabularization.aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
```

### Hyperparameter Optimization

```console
meds-tab-model \
    --multirun \
    model_launcher=xgboost \
    "input_dir=${MEDS_RESHARD_DIR}/data" \
    "output_dir=$OUTPUT_DIR" \
    "output_model_dir=${OUTPUT_MODEL_DIR}/${TASK}/" \
    "task_name=$TASK" \
    "hydra.sweeper.n_trials=1000" \
    "hydra.sweeper.n_jobs=${N_PARALLEL_WORKERS}" \
    tabularization.min_code_inclusion_count=10 \
    tabularization.window_sizes=[1d,30d,365d,full] \
    tabularization.aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
```

??? note "Args Description for Model Stage"
    - `model_launcher`: Choose from `xgboost`, `knn_classifier`, `logistic_regression`, `random_forest_classifier`, `sgd_classifier`
    - `input_dir`: The directory containing the MEDS data
    - `output_dir`: The directory storing tabularized data
    - `output_model_dir`: Where to save model outputs
    - `hydra.sweeper.n_trials`: Number of trials for hyperparameter optimization
    - `hydra.sweeper.n_jobs`: Number of parallel jobs for optimization

??? note "Code Inclusion Parameters in Modeling"
    In this modeling stage, you can change the code inclusion parameters from previous stages and treat them as tunable hyperparameters. Additional task-specific parameters include:

    - `min_correlation`: Minimum correlation with target required
    - `max_by_correlation`: Maximum number of codes to include based on correlation with target

??? note "Data Preprocessing Options"
    - **Tree-based methods** (e.g., XGBoost):
        - Insensitive to normalization
        - Generally don't benefit from missing value imputation
        - XGBoost handles missing data natively
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
        \[Previous structure\]

    ??? folder "MEDS_RESHARD_DIR"
        \[Previous structure\]

    ??? folder "OUTPUT_DIR"
        \[Previous structure\]

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
                    \[Similar structure to TRIAL_1_ID\]

??? example "Experimental Feature"
    We also support an autogluon based hyperparameter and model search:

    ```console
    meds-tab-autogluon model_launcher=autogluon \
       "input_dir=${MEDS_RESHARD_DIR}/data" \
       "output_dir=$OUTPUT_DIR" \
       "output_model_dir=${OUTPUT_MODEL_DIR}/${TASK}/" \
       "task_name=$TASK"
    ```

    Run `meds-tab-autogluon model_launcher=autogluon --help` to see all kwargs. Autogluon requires a lot of memory as it makes all the sparse matrices dense, and is not recommended for large datasets.
