# The MEDS-Tab Architecture

MEDS-Tab is designed to address two key challenges in healthcare machine learning: (1) efficiently tabularizing large-scale electronic health record (EHR) data and (2) training competitive baseline models on this tabularized data. This document outlines the architecture and implementation details of MEDS-Tab's pipeline.

## Overview

The MEDS-Tab pipeline consists of six main stages, with the first (stage 0) being optional:

0. Data Resharding (Optional)
1. Data Description (Code Frequency Analysis)
2. Static Data Tabularization
3. Time-Series Data Tabularization
4. Task-Specific Data Caching
5. Model Training

Each stage is designed with scalability and efficiency in mind, using sparse matrix operations and data sharding to handle large-scale medical datasets.

## Stage 0: Data Resharding (Optional)

This optional preliminary stage helps optimize data processing by restructuring the input data into manageable shards. Resharding is particularly useful when dealing with large datasets or when experiencing memory constraints. The process uses the MEDS_transform-reshard_to_split command and supports parallel processing via Hydra's joblib launcher, with configurable shard sizes based on number of subjects.

Consider resharding if you're experiencing memory issues in later stages, need to process very large datasets, want to enable efficient parallel processing, or have uneven distribution of data across existing shards.

### Output Structure
```text
/PATH/TO/MEDS_RESHARD_DIR
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

## Stage 1: Data Description

The first stage analyzes the MEDS data to compute code frequencies and categorize features. This information is crucial for subsequent feature selection and optimization. The implementation iterates through data shards to compute feature frequencies and categorizes codes into dynamic codes (codes with timestamps), dynamic numeric values (codes with timestamps and numerical values), static codes (codes without timestamps), and static numeric values (codes without timestamps but with numerical values). Results are stored in a `${output_dir}/metadata/codes.parquet` file for use in subsequent stages, where `output_dir` is a key word argument.

### Input Data Structure
```text
/PATH/TO/MEDS/DATA
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

## Stage 2: Static Data Tabularization

This stage processes static patient data (data without timestamps) into a format suitable for modeling. The implementation uses a dense pivot operations which because static data is generally relatively small. Then this stage converts the data to a sparse matrix format for consistency with time-series data. At first there is a single row for each `subject_id` with their static data. This is are duplicated by the number of unique times the patient has data to align with time-series events, and processing over shards is performed serially due to the manageable size of static data.

### Input Data Structure
```text
/PATH/TO/MEDS/DATA
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

### Output Data Structure
```text
${output_dir}/tabularize/
│
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

Note that `.../none/static/present.npz` represents the tabularized data for static features with the aggregation method `static/present`. The `.../none/static/first.npz` represents the tabularized data for static features with the aggregation method `static/first`.

## Stage 3: Time-Series Data Tabularization

This stage handles the computationally intensive task of converting temporal medical data into feature vectors. The process employs several key optimizations: sparse matrix operations utilizing scipy.sparse for memory-efficient storage of sparse non-zero elements, data sharding that processes data in patient-based shards and enables parallel processing, and efficient aggregation using Polars for fast rolling window computations.

The process flow begins by loading shard data into a Polars DataFrame, converting it to sparse matrix format where rows represent events and columns represent features. It then aggregates same-day events per patient, applies rolling window aggregations, and stores results in sparse coordinate format (.npz files).

### Input Data Structure
```text
/PATH/TO/MEDS/DATA
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

### Output Data Structure
```text
${output_dir}/tabularize/
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
    │   ...
```

The output structure consists of a directory for each split, containing subdirectories for each shard. Each shard subdirectory contains subdirectories for each aggregation method and window size, with the final output files stored in sparse coordinate format (.npz). In this example we have shown the output for the `1d` and `7d` window sizes and `code/count` and `value/sum` aggregation methods.

## Stage 4: Task-Specific Data Caching

This stage aligns tabularized data with specific prediction tasks, optimizing for efficient model training. The implementation accepts task labels following the MEDS label-schema and matches them with nearest prior feature vectors. It filters tabularized data to include only task-relevant events while maintaining sparse format for efficient storage. Labels must include subject_id, prediction_time, and boolean_value for binary classification.


### Input Data Structure
```text
${output_dir}/tabularize/ # Output from Stage 2 and 3
${input_label_dir}/**/*.parquet # All parquet files in the `input_label_dir` are used as labels
```


### Output Data Structure

Labels are cached in:
```text
$output_label_cache_dir
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

For each shard, the labels are stored in a parquet file with the same name as the shard. The labels are stored in the `output_label_cache_dir` directory which by default is relative to the key word argument `$output_dir`: `output_label_cache_dir = ${output_dir}/${task_name}/labels`.

Task specific tabularized data is cached in the following format:
```text
$output_tabularized_cache_dir
└─── <SPLIT A>
│   │   <SHARD 0>/1d/code/count.npz
│   │   <SHARD 0>/1d/value/sum.npz
|   |   <SHARD 0>/none/static/present.npz
|   |   <SHARD 0>/none/static/first.npz
|   |   ...
|   |   <SHARD 0>/7d/code/count.npz
│   │   <SHARD 0>/7d/value/sum.npz
│   │   ...
|   |   <SHARD 1>/1d/code/count.npz
│   │   <SHARD 1>/1d/value/sum.npz
│   │   <SHARD 1>/none/static/present.npz
|   |   <SHARD 1>/none/static/first.npz
│   │   ...
│
└─── <SPLIT B>
   │    ...
```
The output structure is identical to the structure in Stages 2 and 3, but where we filter rows in the sparse matrix to only include events relevant to the task. This is done by selecting one row for each label that corresponds with the nearest prior event. The task-specific tabularized data is stored in the `output_tabularized_cache_dir` directory. By default this directory is relative to the key word argument `$output_dir`: `output_tabularized_cache_dir = ${output_dir}/${task_name}/task_cache`.

## Stage 5: Model Training

The final stage provides efficient model training capabilities, particularly optimized for XGBoost. The system incorporates extended memory support through sequential shard loading during training and efficient data loading through custom iterators. AutoML integration uses Optuna for hyperparameter optimization, tuning across model parameters, aggregation methods, window sizes, and feature selection thresholds.

### Input Data Structure
```text
# Location of task, split, and shard specific tabularized data
${input_tabularized_cache_dir} # Output from Stage 4
# Location of  task, split, and shard specific label data
${input_label_cache_dir} # Output from Stage 4
```

### Output Data Structure

For single runs, the output structure is as follows:
```text
# Where to output the model and cached data
time_output_model_dir = ${output_model_dir}/${now:%Y-%m-%d_%H-%M-%S}
├── config.log
├── performance.log
└── xgboost.json # model weights
```

For `multirun` optuna hyperparameter sweeps we get the following output structure:
```text
# Where to output the model and cached data
time_output_model_dir = ${output_model_dir}/${now:%Y-%m-%d_%H-%M-%S}
├── best_trial
|  ├── config.log
|  ├── performance.log
|  └── xgboost.json # model weights
├── hydra
|  └── optimization_results.yaml # contains the optimal trial hyperparameters and performance
└── sweep_results # This folder contains raw results for every hyperparameter trial
   └── <TRIAL_1_ID>
      ├── config.log # model config log
      ├── performance.log # model performance log
      └── xgboost.json # model weights
   └── <TRIAL_2_ID>
   ...
```

`output_model_dir` is a keyword argument that specifies the directory where the model and cached data are stored. By default, we append the current date and time to the directory name to avoid overwriting previous runs, and use the `time_output_model_dir` variable to store the full path. If you use a different `model_launcher` than XGBoost, the model weights file will be named accordingly for that model (and will be a `.pkl` file instead of a `json`).

### Supported Models and Processing Options
The default model is XGBoost, with additional options including KNN Classifier, Logistic Regression, Random Forest Classifier, SGD Classifier, and experimental AutoGluon support. Data processing options include sparse-preserving normalization (standard_scaler, max_abs_scaler) and imputation methods that convert to dense format (mean_imputer, median_imputer, mode_imputer). By default no normalization is applied and missing values are treated as missing by `xgboost` or as zero by other models.

## Additional Considerations

The architecture emphasizes robust memory management through sparse matrices and efficient data sharding, while supporting parallel processing and handling of high-dimensional feature spaces. The system is optimized for performance, minimizing memory footprint and computational overhead while enabling processing of datasets with hundreds of millions of events and tens of thousands of unique medical codes.
