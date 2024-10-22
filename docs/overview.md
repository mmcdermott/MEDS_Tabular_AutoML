# Core CLI Scripts Overview

We provide a set of core CLI scripts to facilitate the tabularization and modeling of MEDS data. These scripts are designed to be run in sequence to transform raw MEDS data into tabularized data and train a model on the tabularized data. The following is a high-level overview of the core CLI scripts:

#### 1. **`MEDS_transform-reshard_to_split`**:

This optional command reshards the data. A core challenge in tabularization is the high memory usage and slow compute time. We shard the data into small shards to reduce the memory usage as we can independently tabularize each shard, and we can reduce cpu time by parallelizing the processing of these shards across workers that are independently processing different shards.

```bash
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

```
- `--multirun`: This is an optional argument to specify that the command should be run in parallel. We use this here to parallelize the resharing of the data.
- `hydra/launcher`: This is an optional argument to specify the launcher. When using multirun you should specify the launcher. We use joblib here which enables parallelization on a single machine.
- `worker`: When using joblib or a hydra slurm launcher, the range of workers must be defined as it specifies the number of parallel workers to spawn. We use 6 workers here.
- `input_dir`: The directory containing the MEDS data.
- `cohort_dir`: The directory to store the resharded data.
- `stages`: The stages to run. We only run the reshard_to_split stage here. MEDS Transform allows for a sequence of stages to be defined an run which is why this is a list.
- `stage`: The specific stage to run. We run the reshard_to_split stage here. It must be one of the stages in the `stages` kwarg list.
- `stage_configs.reshard_to_split.n_subjects_per_shard`: The number of subjects per shard. We use 2500 subjects per shard here.
```

For the rest of the tutorial we will assume that the data has been reshared into the `MEDS_RESHARD_DIR` directory, but this step is optional, and you could instead use the original data directory, `MEDS_DIR`. If you experience high memory issues in later stages, you should try reducing `stage_configs.reshard_to_split.n_subjects_per_shard` to a smaller number.

#### 2. **`meds-tab-describe`**:

This command processes MEDS data shards to compute the frequencies of different code types. It differentiates codes into the following categories:

- dynamic codes (codes with timestamps)

- dynamic numeric values (codes with timestamps and numerical values)

- static codes (codes without timestamps)

- static numeric values (codes without timestamps but with numerical values).

  This script further caches feature names and frequencies in a dataset stored in a `code_metadata.parquet` file within the `input_dir` argument specified as a hydra-style command line argument.

```bash
meds-tab-describe \
    "input_dir=${MEDS_RESHARD_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR"
```

This stage is not parallelized as it runs very quickly.

??? note "Args Description"

```
- `input_dir`: The directory containing the MEDS data.
- `output_dir`: The directory to store the tabularized data.
```

#### 3. **`meds-tab-tabularize-static`**: Filters and processes the dataset based on the count of codes, generating a tabular vector for each patient at each timestamp in the shards. Each row corresponds to a unique `subject_id` and `timestamp` combination, thus rows are duplicated across multiple timestamps for the same patient.

**Example: Tabularizing static data** with the minimum code count of 10, window sizes of `[1d, 30d,  365d, full]`, and value aggregation methods of `[static/present, static/first, code/count, value/count, value/sum, value/sum_sqd, value/min, value/max]`

```console
meds-tab-tabularize-static input_dir="path_to_data" \
                            tabularization.min_code_inclusion_count=10 \
                            tabularization.window_sizes=[1d,30d,365d,full] \
                            do_overwrite=False \
                            tabularization.aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]"
```

- For the exhaustive examples of value aggregations, see [`/src/MEDS_tabular_automl/utils.py`](https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/main/src/MEDS_tabular_automl/utils.py#L24)

!!! note

```
In addition to `min_code_inclusion_count` there are several other parameters that can be set tabularization to restrict the codes that are included in the tabularized data. These are:
* `allowed_codes`: a list of codes to include in the tabularized data
* `min_code_inclusion_count`: The minimum number of times a code must appear in the data to be included in the tabularized data
* `min_code_inclusion_frequency` The minimum normalized frequency (i.e. normalized by dividing the code's count by the total number of observations across all codes in the dataset) required for a code to be included.
* `max_included_codes`: The maximum number of codes to include in the tabularized data
```

```bash
meds-tab-tabularize-static \
    "input_dir=${MEDS_RESHARD_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR" \
```

This stage is not parallelized as it runs very quickly.
??? note "Args Description"
\- `input_dir`: The directory containing the MEDS data.
\- `output_dir`: The directory to store the tabularized data.

#### 4. **`meds-tab-tabularize-time-series`**:

Iterates through combinations of a shard, `window_size`, and `aggregation` to generate feature vectors that aggregate patient data for each unique `subject_id` x `time`. This stage (and the previous stage) uses sparse matrix formats to efficiently handle the computational and storage demands of rolling window calculations on large datasets. We support parallelization through Hydra's [`--multirun`](https://hydra.cc/docs/intro/#multirun) flag and the [`joblib` launcher](https://hydra.cc/docs/plugins/joblib_launcher/#internaldocs-banner).

**Example: Aggregate time-series data** on features across different `window_sizes`

```bash
meds-tab-tabularize-time-series \
    --multirun \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    hydra/launcher=joblib \
    "input_dir=${MEDS_RESHARD_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR" \
    tabularization.min_code_inclusion_count=10 \
    tabularization.window_sizes=[1d,30d,365d,full] \
    tabularization.aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
```

!!! warning

```
This stage is the most memory intensive stage! This stage should be parallelized to speed up the processing of the data. If you run out of memory, either reduce the workers or reshard your data with `MEDS_transform-reshard_to_split` setting `stage_configs.reshard_to_split.n_subjects_per_shard` to a smaller number. This stage is also one of the the slowest stages.
```

!!! warning

```
You must use the same code inclusion parameters (which in this example  is just `tabularization.min_code_inclusion_count`) as in the previous stage, `meds-tab-tabularize-static`, to ensure that the same codes are included in the tabularized data.
```

??? note "Args Description"

```
- `--multirun`: This is an optional argument to specify that the command should be run in parallel. We use this here to parallelize the resharing of the data.
- `hydra/launcher`: This is an optional argument to specify the launcher. When using multirun you should specify the launcher. We use joblib here which enables parallelization on a single machine.
- `worker`: When using joblib or a hydra slurm launcher, the range of workers must be defined as it specifies the number of parallel workers to spawn. We use `$N_PARALLEL_WORKERS` workers here.
- `input_dir`: The directory containing the MEDS data.
- `output_dir`: The directory to store the tabularized data.
- `tabularization.min_code_inclusion_count`: The minimum code inclusion frequency. We use 10 here, so only codes that appear at least 10 times in the data will be included.
- `tabularization.window_sizes`: The window sizes to use. We use `[1d,30d,365d,full]` here. This means we will generate features for the last day, last 30 days, last 365 days, and the full history of the patient.
- `tabularization.aggs`: The aggregation functions to use. We use `[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]` here. This means we will generate features for the presence of a static code, the value of a static code, the count of dynamic codes, the count of dynamic values, the sum of dynamic values, the sum of squared dynamic values, the minimum dynamic value, and the maximum dynamic value.
```

5. **`meds-tab-cache-task`**:

Aligns task-specific labels with the nearest prior event in the tabularized data. It requires a labeled dataset directory with three columns (`subject_id`, `timestamp`, `label`) structured similarly to the `input_dir`.

**Example: Align tabularized data** for a specific task `$TASK` and labels that has pulled from [ACES](https://github.com/justin13601/ACES)

```console
meds-tab-cache-task input_dir="path_to_data" \
   task_name=$TASK \
   tabularization.min_code_inclusion_count=10 \
   tabularization.window_sizes=[1d,30d,365d,full] \
   tabularization.aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
```

```bash
meds-tab-cache-task \
    --multirun \
    hydra/launcher=joblib \
    worker="range(0,$N_PARALLEL_WORKERS)" \
    "input_dir=${MEDS_RESHARD_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR" \
    "input_label_dir=${TASKS_DIR}/${TASK}/" "task_name=${TASK}"
    tabularization.min_code_inclusion_count=10 \
    tabularization.window_sizes=[1d,30d,365d,full] \
    tabularization.aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
```

!!! warning

```
This stage is the slowest stage, but should not be as memory intensive, so make sure to parallelize across as many workers as possible.
```

!!! warning

```
You must use the same code inclusion parameters (which in this example  is just `tabularization.min_code_inclusion_count`) as in the previous stages, `meds-tab-tabularize-static` and `meds-tab-tabularize-time-series`, to ensure that the same codes are included in the tabularized data.
```

??? note "Args Description"

```
- `--multirun`: This is an optional argument to specify that the command should be run in parallel. We use this here to parallelize the resharing of the data.
- `hydra/launcher`: This is an optional argument to specify the launcher. When using multirun you should specify the launcher. We use joblib here which enables parallelization on a single machine.
- `worker`: When using joblib or a hydra slurm launcher, the range of workers must be defined as it specifies the number of parallel workers to spawn. We use `$N_PARALLEL_WORKERS` workers here.
- `input_dir`: The directory containing the MEDS data.
- `output_dir`: The directory to store the tabularized data.
- `input_label_dir`: The directory containing the labels (following the [meds label-schema](https://github.com/Medical-Event-Data-Standard/meds?tab=readme-ov-file#the-label-schema)) for the task.
- `task_name`: The name of the task to cache the labels for.
- `tabularization.min_code_inclusion_count`: The minimum code inclusion frequency.
- `tabularization.window_sizes`: The window sizes to use.
- `tabularization.aggs`: The aggregation functions to use.
```

#### 6. **`meds-tab-model`**:

Trains a tabular model using user-specified parameters. You can train a single xgboost model with the following command:

```bash
meds-tab-model \
    model_launcher=xgboost \
    "input_dir=${MEDS_RESHARD_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR" \
    "output_model_dir=${OUTPUT_MODEL_DIR}/${TASK}/" "task_name=$TASK" \
    tabularization.min_code_inclusion_count=10 \
    "tabularization.window_sizes=[1d,30d,365d,full]" \
    "tabularization.aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]"
```

??? note "Args Description"
\- `model_launcher`: The launcher to use for the model. choose one in `xgboost`, `knn_classifier`, `logistic_regression`, `random_forest_classifier`, `sgd_classifier`.
\- `input_dir`: The directory containing the MEDS data.
\- `output_dir`: The directory to store the tabularized data.
\- `output_model_dir`: The directory to store the model.
\- `hydra.sweeper.n_trials`: The number of trials to run in the hyperparameter sweep.
\- `hydra.sweeper.n_jobs`: The number of parallel jobs to run in the hyperparameter sweep.
\- `task_name`: The name of the task to cache the labels for.
\- `tabularization.min_code_inclusion_count`: The minimum code inclusion frequency.
\- `tabularization.window_sizes`: The window sizes to use.
\- `tabularization.aggs`: The aggregation functions to use.

You can also run an [optuna](https://optuna.org/) hyperparameter sweep by adding the `--multirun` flag and can control the number of trials with `hydra.sweeper.n_trials` and parallel jobs with `hydra.sweeper.n_jobs`:

```bash
meds-tab-model \
   --multirun \
   model_launcher=xgboost \
   "input_dir=${MEDS_RESHARD_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR" \
   "output_model_dir=${OUTPUT_MODEL_DIR}/${TASK}/" "task_name=$TASK" \
   "hydra.sweeper.n_trials=1000" "hydra.sweeper.n_jobs=${N_PARALLEL_WORKERS}" \
    tabularization.min_code_inclusion_count=10 \
    tabularization.window_sizes=$(generate-subsets [1d,30d,365d,full]) \
    tabularization.aggs=$(generate-subsets [static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max])
```

??? note "Args Description"
\- `multirun`: This is a required argument when sweeping and specifies that we are performing a hyperparameter sweep and using optuna.
\- `model_launcher`: The launcher to use for the model. choose one in `xgboost`, `knn_classifier`, `logistic_regression`, `random_forest_classifier`, `sgd_classifier`.
\- `input_dir`: The directory containing the MEDS data.
\- `output_dir`: The directory to store the tabularized data.
\- `output_model_dir`: The directory to store the model.
\- `hydra.sweeper.n_trials`: The number of trials to run in the hyperparameter sweep.
\- `hydra.sweeper.n_jobs`: The number of parallel jobs to run in the hyperparameter sweep.
\- `task_name`: The name of the task to cache the labels for.
\- `tabularization.min_code_inclusion_count`: The minimum code inclusion frequency.
\- `tabularization.window_sizes`: The window sizes to use.
\- `tabularization.aggs`: The aggregation functions to use.

??? note "Why `generate-subsets`?"
**`generate-subsets`**: Generates and prints a sorted list of all non-empty subsets from a comma-separated input. This is provided for the convenience of sweeping over all possible combinations of window sizes and aggregations.

````
For example, you can directly call **`generate-subsets`** in the command line:

```console
generate-subsets [2,3,4] \
[2], [2, 3], [2, 3, 4], [2, 4], [3], [3, 4], [4]
```

This could be used in the command line in concert with other calls. For example, the following call:

```console
meds-tab-model --multirun tabularization.window_sizes=$(generate-subsets [1d,2d,7d,full])
```

would resolve to:

```console
meds-tab-model --multirun tabularization.window_sizes=[1d],[1d,2d],[1d,2d,7d],[1d,2d,7d,full],[1d,2d,full],[1d,7d],[1d,7d,full],[1d,full],[2d],[2d,7d],[2d,7d,full],[2d,full],[7d],[7d,full],[full]
```

which can then be correctly interpreted by Hydra's multirun logic to sweep over all possible combinations of window sizes, during hyperparameter tuning!
````

!!! note "Code Inclusion Parameters"

```
In this modeling stage, you can change the code inclusion parameters from the previous tabularization and task caching stages, and treat them as a tunable hyperparameter

In addition to the previously defined code inclusion parameters, there are two others that we allow only in modeling (as they are task specific):
* `min_correlation`: The minimum correlation a code must have with the target to be included in the tabularized data
* `max_by_correlation`: The maximum number of codes to include in the tabularized data based on correlation with the target. Specifically we sort the codes by correlation with the target and include the top `max_by_correlation` codes.
```

??? example "Experimental Feature"

````
We also support an autogluon based hyperparameter and model search:
```bash
meds-tab-autogluon model_launcher=autogluon \
   "input_dir=${MEDS_RESHARD_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR" \
   "output_model_dir=${OUTPUT_MODEL_DIR}/${TASK}/" "task_name=$TASK" \
```
run `meds-tab-autogluon model_launcher=autogluon --help` to see all kwargs. Autogluon requires a lot of memory as it makes all the sparse matrices dense, and is not recommended for large datasets.
````
