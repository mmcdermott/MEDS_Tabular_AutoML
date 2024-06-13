# Scalable tabularization and tabular feature usage utilities over generic MEDS datasets

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3100/"><img alt="Python" src="https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white"></a>
  <a href="https://pypi.org/project/meds-tab/"><img alt="PyPI" src="https://img.shields.io/badge/PyPI-v0.2.5-orange?logoColor=orange"></a>
  <a href="https://hydra.cc/"><img alt="Hydra" src="https://img.shields.io/badge/Config-Hydra_1.3-89b8cd"></a>
  <a href="https://codecov.io/gh/mmcdermott/MEDS_Tabular_AutoML"><img alt="Codecov" src="https://codecov.io/gh/mmcdermott/MEDS_Tabular_AutoML/graph/badge.svg?token=6GD05EDQ39"></a>
  <a href="https://github.com/mmcdermott/MEDS_Tabular_AutoML/actions/workflows/tests.yaml"><img alt="Tests" src="https://github.com/mmcdermott/MEDS_Tabular_AutoML/actions/workflows/tests.yaml/badge.svg"></a>
  <a href="https://github.com/mmcdermott/MEDS_Tabular_AutoML/actions/workflows/code-quality-main.yaml"><img alt="Code Quality" src="https://github.com/mmcdermott/MEDS_Tabular_AutoML/actions/workflows/code-quality-main.yaml/badge.svg"></a>
  <a href='https://meds-tab.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/meds-tab/badge/?version=latest' alt='Documentation Status' /></a>
  <a href="https://github.com/mmcdermott/MEDS_Tabular_AutoML/graphs/contributors"><img alt="Contributors" src="https://img.shields.io/github/contributors/mmcdermott/MEDS_Tabular_AutoML.svg"></a>
  <a href="https://github.com/mmcdermott/MEDS_Tabular_AutoML/pulls"><img alt="Pull Requests" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
  <a href="https://github.com/mmcdermott/MEDS_Tabular_AutoML#license"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray"></a>
</p>

This repository provides utilities and scripts to run limited automatic tabular ML pipelines for generic MEDS
datasets.

# Installation

To use MEDS-Tab, install the dependencies following commands below:

**Pip Install**

```bash
pip install meds-tab
```

**Local Install**

```
# clone the git repo
pip install .
```

# Usage

This repository consists of two key pieces:

1. Construction of and efficient loading of tabular (flat, non-longitudinal) summary features describing patient records in MEDS over arbitrary time-windows (e.g. 1 year, 6 months, etc.) backwards in time from a given index date.
2. Running a basic XGBoost AutoML pipeline over these tabular features to predict arbitrary binary classification or regression downstream tasks defined over these datasets. The "AutoML" part of this is not particularly advanced -- what is more advanced is the efficient construction, storage, and loading of tabular features for the candidate AutoML models, enabling a far more extensive search over a much larger total number of features than prior systems.

## Quick Start

To use MEDS-Tab, install the dependencies following commands below:

**Pip Install**

```bash
pip install meds-tab
```

**Local Install**

```bash
# clone the git repo
pip install .
```

## Scripts and Examples

For an end to end example over MIMIC-IV, see the [MIMIC-IV companion repository](https://github.com/mmcdermott/MEDS_TAB_MIMIC_IV).
For an end to end example over Philips eICU, see the [eICU companion repository](https://github.com/mmcdermott/MEDS_TAB_EICU).

See `tests/test_integration.py` for a local example of the end-to-end pipeline being run on synthetic data. This script is a functional test that is also run with `pytest` to verify the correctness of the algorithm.

## Core CLI Scripts Overview

1. **`meds-tab-describe`**: This command processes MEDS data shards to compute the frequencies of different code-types. It differentiates codes into the following categories:

   - time-series codes (codes with timestamps)
   - time-series numerical values (codes with timestamps and numerical values)
   - static codes (codes without timestamps)
   - static numerical codes (codes without timestamps but with numerical values).

   This script further caches feature names and frequencies in a dataset stored in a `code_metadata.parquet` file within the `MEDS_cohort_dir` argument specified as a hydra-style command line argument.

2. **`meds-tab-tabularize-static`**: Filters and processes the dataset based on the frequency of codes, generating a tabular vector for each patient at each timestamp in the shards. Each row corresponds to a unique `patient_id` and `timestamp` combination, thus rows are duplicated across multiple timestamps for the same patient.

   **Example: Tabularizing static data** with the minimum code frequency of 10 and window sizes of `[1d, 30d,  365d, full]` and value aggregation methods of `[static/present, code/count, value/count, value/sum, value/sum_sqd, value/min, value/max]`

   ```bash
   meds-tab-tabularize-static MEDS_cohort_dir="path_to_data" \
                               tabularization.min_code_inclusion_frequency=10 \
                               tabularization.window_sizes=[1d,30d,365d,full] \
                               do_overwrite=False \
                               tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]"
   ```

   - For the exhuastive examples of value aggregations, see [`/src/MEDS_tabular_automl /utils.py`](https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/main/src/MEDS_tabular_automl/utils.py#L24)

3. **`meds-tab-tabularize-time-series`**: Iterates through combinations of a shard, `window_size`, and `aggregation` to generate feature vectors that aggregate patient data for each unique `patient_id` x `timestamp`. This stage (and the previous stage) use sparse matrix formats to efficiently handle the computational and storage demands of rolling window calculations on large datasets. We support parallelization through Hydra's [`--multirun`](https://hydra.cc/docs/intro/#multirun) flag and the [`joblib` launcher](https://hydra.cc/docs/plugins/joblib_launcher/#internaldocs-banner).

   **Example: Aggregate time-series data** on features across different `window_sizes`

   ```bash
   meds-tab-tabularize-time-series --multirun \
                                   worker="range(0,$N_PARALLEL_WORKERS)" \
                                   hydra/launcher=joblib \
                                   MEDS_cohort_dir="path_to_data" \
                                   tabularization.min_code_inclusion_frequency=10 \
                                   do_overwrite=False \
                                   tabularization.window_sizes=[1d,30d,365d,full] \
                                   tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
   ```

4. **`meds-tab-cache-task`**: Aligns task-specific labels with the nearest prior event in the tabularized data. It requires a labeled dataset directory with three columns (`patient_id`, `timestamp`, `label`) structured similarly to the `MEDS_cohort_dir`.

   **Example: Aligh tabularized data** for a specific task `$TASK` and labels that has pulled from [ACES](https://github.com/justin13601/ACES)

   ```bash
   meds-tab-cache-task MEDS_cohort_dir="path_to_data" \
                       task_name=$TASK \
                       tabularization.min_code_inclusion_frequency=10 \
                       do_overwrite=False \
                       tabularization.window_sizes=[1d,30d,365d,full] \
                       tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
   ```

5. **`meds-tab-xgboost`**: Trains an XGBoost model using user-specified parameters. Permutations of `window_sizes` and `aggs` can be generated using `generate-permutations` command (See the section below for descriptions).

   ```bash
   meds-tab-xgboost --multirun \
                     MEDS_cohort_dir="path_to_data" \
                     task_name=$TASK \
                     output_dir="output_directory" \
                     tabularization.min_code_inclusion_frequency=10 \
                     tabularization.window_sizes=$(generate-permutations [1d,30d,365d,full]) \
                     do_overwrite=False \
                     tabularization.aggs=$(generate-permutations [static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max])
   ```

## Additional CLI Scripts

1. **`generate-permutations`**: Generates and prints a sorted list of all permutations from a comma separated input. This is provided for the convenience of sweeping over all possible combinations of window sizes and aggregations.

   For example you can directly call **`generate-permutations`** in the command line:

   ```bash
   generate-permutations [2,3,4]
   [2], [2, 3], [2, 3, 4], [2, 4], [3], [3, 4], [4]
   ```

   This could be used in the command line in concert with other calls. For example, the following call:

   ```bash
   meds-tab-xgboost --multirun tabularization.window_sizes=$(generate-permutations [1d,2d,7d,full])
   ```

   would resolve to:

   ```bash
   meds-tab-xgboost --multirun tabularization.window_sizes=[1d],[1d,2d],[1d,2d,7d],[1d,2d,7d,full],[1d,2d,full],[1d,7d],[1d,7d,full],[1d,full],[2d],[2d,7d],[2d,7d,full],[2d,full],[7d],[7d,full],[full]
   ```

   which can then be correctly interpreted by Hydra's multirun logic.

## Roadmap

MEDS-Tab has several key limitations which we plan to address in future changes. These include, and are tracked by, the following GitHub issues.

### Improvements to the core tabularization

1. Further memory and runtime improvements are possible: [#16](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/16)
2. We should support additional window sizes and types: [#31](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/31)
3. We should support additional aggregation functions: [#32](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/32)

### Improvements to the modeling pipeline

1. We should likely decorrelate the default aggregations and/or window sizes we use prior to passing them into the models as features: [#27](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/27)
2. We need to do a detailed parameter study over the hyperparameter sweep options to find good defaults for these kinds of problems and models: [#33](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/33)
3. We should support a more extensive set of pipeline operations and model architectures: [#37](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/37)

### Technical debt / code improvements

1. The computation and use of the code metadata dataframe, containing frequencies of codes, should be offloaded to core MEDS functionality, with the remaining code in this repository cleaned up.
   - [#28](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/28)
   - [#14](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/14)
2. We should add more doctests and push test coverage up to 100%
   - [#29](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/29)
   - [#30](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/30)
3. We need to ensure full and seamless compatibility with the ACES CLI tool, rather than relying on the python API and manual adjustments:
   [#34](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/34)

# How does MEDS-Tab Work?

## What do you mean "tabular pipelines"? Isn't _all_ structured EHR data already tabular?

This is a common misconception. _Tabular_ data refers to data that can be organized in a consistent, logical
set of rows/columns such that the entirety of a "sample" or "instance" for modeling or analysis is contained
in a single row, and the set of columns possibly observed (there can be missingness) is consistent across all
rows. Structured EHR data does not satisfy this definition, as we will have different numbers of observations
of medical codes and values at different timestamps for different patients, so it cannot simultanesouly
satisfy the (1) "single row single instance", (2) "consistent set of columns", and (3) "logical" requirements.
Thus, in this pipeline, when we say we will produce a "tabular" view of MEDS data, we mean a dataset that can
realize these constraints, which will explicitly involve summarizing the patient data over various historical
or future windows in time to produce a single row per patient with a consistent, logical set of columns
(though there may still be missingness).

## The MEDS-Tab Architecture

In this section we describe the MEDS-Tab architecture, specifically some of the pipeline choices we made to reduce memory usage and increase speed during the tabularization process and XGBoost tuning process.

We break our method into 4 discrete parts

1. Describe codes (compute feature frequencies)
2. Given time series data tabularize it
3. cache task specific rows of data for efficient loading
4. XGBoost training

### 1. Describe Codes (Compute Feature Frequencies)

This initial stage processes a pre-shareded dataset. We expect a structure as follows where each shard contains a subset of the patients:

```
/PATH/TO/MEDS/DATA
│
└───<SPLIT A>
│   │   <SHARD 0>.parquet
│   │   <SHARD 1>.parquet
│   │   ...
│
└───<SPLIT B>
│   │   <SHARD 0>.parquet
│   │   <SHARD 1>.parquet
|   │   ...
|
...
```

We then compute and store feature frequencies, crucial for determining which features are relevant for further analysis.

**Detailed Workflow:**

- **Data Loading and Sharding**: We iterate through shards to compute feature frequencies for each shard.
- **Frequency Aggregation**: After computing frequencies across shards, we aggregate them to get a final count of each feature across the entire dataset training dataset, which allows us to filter out infrequent features in the tabularization stage or when tuning XGBoost.

This outputs parquet file \`\`

### 2. Tabularization of Time Series Data

### Overview

The tabularization stage of our pipeline, exposed via the cli commands:

- `meds-tab-tabularize-static` for tabularizing static data
- and `meds-tab-tabularize-time-series` for tabularizing the time series data

Static data is relatively small in the medical datasets, so we use a dense pivot operation, convert it to a sparse matrix, and then duplicate rows such that the static data will match up with the time series data rows generated in the next step. Static data is currently processed serially.

The script for tabularizing time series data primarily transforms a raw, unstructured dataset into a structured, feature-rich dataset by utilizing a series of sophisticated data processing steps. This transformation involves converting raw time series from a Polars dataframe into a sparse matrix format, aggregating events that occur at the same date for the same patient, and then applying rolling window aggregations to extract temporal features. Here's a step-by-step breakdown of the algorithm:

### High-Level Steps

1. **Data Loading and Categorization**:

   - The script iterates through shards of patients, and shards can be processed in parallel using hydras joblib to launch multiple processes.

2. **Sparse Matrix Conversion**:

   - Data from the Polars dataframe is converted into a sparse matrix format. This step is crucial for efficient memory management, especially when dealing with large datasets.

3. **Event Aggregation**:

   - Events that occur on the same date for the same patient are aggregated. This reduces redundancy in the data and significantly speeds up the rolling window aggregations on datasets that have lots of concurrent observations.

4. **Rolling Window Aggregation**:

   - The aggregated data undergoes a rolling window operation where various statistical methods are applied (sum, count, min, max, etc.) to extract features over specified window sizes.

5. **Output Storage**:

   - Sparse array is converted to Coordinate List (COO) format and stored as a `.npz` file on disk.
   - The file paths look as follows

```
/PATH/TO/MEDS/TABULAR_DATA
│
└───<SPLIT A>
    ├───<SHARD 0>
    │   ├───code
    │   │   └───count.npz
    │   └───value
    │       └───sum.npz
    ...
```

### 3. Efficient Data Caching for Task-Specific Rows

Now that we have generated tabular features for all the events in our dataset, we can cache subsets relevant for each task we wish to train a supervised model on. This step is critical for efficiently training machine learning models on task-specific data without having to load the entire dataset.

**Detailed Workflow:**

- **Row Selection Based on Tasks**: Only the data rows that are relevant to the specific tasks are selected and cached. This reduces the memory footprint and speeds up the training process.
- **Use of Sparse Matrices for Efficient Storage**: Sparse matrices are again employed here to store the selected data efficiently, ensuring that only non-zero data points are kept in memory, thus optimizing both storage and retrieval times.

The file structure for the cached data mirrors the tabular data and alsi is `.npz` files, and users must specify the directory to labels that follow the same shard filestructure as the input meds data from step (1). Label parquets need `patient_id`, `timestamp`, and `label` columns.

### 4. XGBoost Training

The final stage uses the processed and cached data to train an XGBoost model. This stage is optimized to handle the sparse data structures produced in earlier stages efficiently.

**Detailed Workflow:**

- **Iterator for Data Loading**: Custom iterators are designed to load sparse matrices efficiently into the XGBoost training process, which can handle sparse inputs natively, thus maintaining high computational efficiency.
- **Training and Validation**: The model is trained using the tabular data, with evaluation steps that include early stopping to prevent overfitting and tuning of hyperparameters based on validation performance.
- **Hyperaparameter Tuning**: We use [optuna](https://optuna.org/) to tune over XGBoost model pramters, aggregations, window_sizes, and the minimimu code inclusion frequency.

# Computational Performance vs. Existing Pipelines

# XGBoost Performance

## XGBoost Model Performance on MIMIC-IV

## XGBoost Model Performance on Philips eICU
