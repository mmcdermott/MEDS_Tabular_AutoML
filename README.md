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

```console
pip install meds-tab
```

**Local Install**

```console
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

```console
pip install meds-tab
```

**Local Install**

```console
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

   ```console
   meds-tab-tabularize-static MEDS_cohort_dir="path_to_data" \
       tabularization.min_code_inclusion_frequency=10 \
       tabularization.window_sizes=[1d,30d,365d,full] \
       do_overwrite=False \
       tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
   ```

   - For the exhuastive examples of value aggregations, see [`/src/MEDS_tabular_automl /utils.py`](https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/main/src/MEDS_tabular_automl/utils.py#L24)

3. **`meds-tab-tabularize-time-series`**: Iterates through combinations of a shard, `window_size`, and `aggregation` to generate feature vectors that aggregate patient data for each unique `patient_id` x `timestamp`. This stage (and the previous stage) use sparse matrix formats to efficiently handle the computational and storage demands of rolling window calculations on large datasets. We support parallelization through Hydra's [`--multirun`](https://hydra.cc/docs/intro/#multirun) flag and the [`joblib` launcher](https://hydra.cc/docs/plugins/joblib_launcher/#internaldocs-banner).

   **Example: Aggregate time-series data** on features across different `window_sizes`

   ```console
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

   ```console
   meds-tab-cache-task MEDS_cohort_dir="path_to_data" \
       task_name=$TASK \
       tabularization.min_code_inclusion_frequency=10 \
       do_overwrite=False \
       tabularization.window_sizes=[1d,30d,365d,full] \
       tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
   ```

5. **`meds-tab-xgboost`**: Trains an XGBoost model using user-specified parameters. Permutations of `window_sizes` and `aggs` can be generated using `generate-permutations` command (See the section below for descriptions).

   ```console
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

   ```console
   generate-permutations [2,3,4]
   [2], [2, 3], [2, 3, 4], [2, 4], [3], [3, 4], [4]
   ```

   This could be used in the command line in concert with other calls. For example, the following call:

   ```console
   meds-tab-xgboost --multirun tabularization.window_sizes=$(generate-permutations [1d,2d,7d,full])
   ```

   would resolve to:

   ```console
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

   - Sparse array is converted to Coordinate List format and stored as a `.npz` file on disk.
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

Evaluating the computational overhead of tabularization methods is essential for assessing their efficiency and suitability for large-scale medical data processing. This section presents a comparative analysis of the computational overhead of MEDS-Tab with other systems like Catabra and TSFresh. It outlines the performance of each system in terms of wall time, memory usage, and output size, highlighting the computational efficiency and scalability of MEDS-Tab.

______________________________________________________________________

## 1. System Comparison Overview

The systems compared in this study represent different approaches to data tabularization, with the main difference being MEDS-Tab usage of sparse tabularization. Specifically, for comparison we used:

1. **Catabra/Catabra-Mem**: Offers data processing capabilities for time-series medical data, with variations to test memory management.
2. **TSFresh**: Both known and used for extensive feature extraction capabilities.

The benchmarking tests were conducted using the following hardware and software settings:

- **CPU Specification**: 2 x AMD EPYC 7713 64-Core Processor
- \*\*RAM Specification: 1024GB, 3200MHz, DDR4
- **Software Environment**: Ubuntu 22.04.4 LTS

## MEDS-Tab Tabularization Technique

Tabularization of time-series data, as depecited above, is commonly used in several past works. The only two libraries to our knowledge that provide a full tabularization pipeline are `tsfresh` and `catabra`. `catabra` also offers a slower but more memory efficient version of their method which we denote `catabra-mem`. Other libraries either provide only rolling window functionalities (`featuretools`) or just pivoting operations (`Temporai`/`Clairvoyance`, `sktime`, `AutoTS`). We provide a significantly faster and more memory efficient method. We find that on the MIMICIV and EICU medical datasets we significantly outperform past methods. `catabra` and `tsfresh` could not even run within a budget of 10 minutes on as low as 10 patient's data for EICU, while our method can scale to process hundreds of patients with low memory usage. We present the results below.

## 2. Comparative Performance Analysis

The tables below detail computational resource utilization across two datasets and various patient scales, emphasizing the better performance of MEDS-Tab in all of the scenarios. The tables are organized by dataset and number of patients. For the analysis, the full window sizes and the aggregation method code_count were used. We additionally use a budget of 10 minutes as these are very small number of patients (10, 100, and 500 patients), and should be processed quickly. Note that `catabra-mem` is omitted from the tables as it never completed within the 10 minute budget.

## eICU Dataset

______________________________________________________________________

The only method that was able to tabularize eICU data was MEDS-Tab. We ran our method with both 100 and 500 patients, resulting in an increment by three times in the number of codes. MEDS-Tab gave efficient results in terms of both time and memory usage.

a) 100 Patients

**Table 1: 6,374 Codes, 2,065,608 Rows, Output Shape \[132,461, 6,374\]**

| Wall Time | Avg Memory  | Peak Memory  | Output Size | Method   |
| --------- | ----------- | ------------ | ----------- | -------- |
| 0m39.426s | 5,271.85 MB | 14,791.83 MB | 362 MB      | meds_tab |

b) 500 Patients

**Table 2: 18,314 Codes, 8,737,355 Rows, Output Shape \[565,014, 18,314\]**

| Wall Time | Avg Memory  | Peak Memory  | Output Size | Method   |
| --------- | ----------- | ------------ | ----------- | -------- |
| 3m4.435s  | 8,335.44 MB | 15,102.55 MB | 1,326 MB    | meds_tab |

## MIMIC-IV Dataset

______________________________________________________________________

MEDS-Tab, TSFresh, and Catabra were tested across three different scales on MIMIC_IV.

a) 10 Patients

This table illustrates the efficiency of MEDS-Tab in processing a small subset of patients with extremely low computational cost and high data throughput, outperforming TSFresh and Catabra in terms of both time and memory efficiency.

**Table 3: 1,504 Codes, 23,346 Rows, Output Shape \[2,127, 1,504\]**

| Wall Time | Avg Memory   | Peak Memory   | Output Size | Method   |
| --------- | ------------ | ------------- | ----------- | -------- |
| 0m2.071s  | 423.78 MB    | 943.984 MB    | 7 MB        | meds_tab |
| 1m41.920s | 84,159.44 MB | 265,877.86 MB | 1 MB        | tsfresh  |
| 0m15.366s | 2,537.46 MB  | 4,781.824 MB  | 1 MB        | catabra  |

b) 100 Patients

For a moderate patient count, MEDS-Tab demonstrated superior performance with significantly lower wall times and memory usage compared to TSFresh and Catabra. The performance gap was further highlighted with an increased number of patients and codes.

**Table 4: 4,154 Codes, 150,789 Rows, Output Shape \[15,664, 4,154\]**

| Wall Time | Avg Memory    | Peak Memory   | Output Size | Method   |
| --------- | ------------- | ------------- | ----------- | -------- |
| 0m4.724s  | 718.76 MB     | 1,167.29 MB   | 45 MB       | meds_tab |
| 5m9.077s  | 217,477.52 MB | 659,735.25 MB | 4 MB        | tsfresh  |
| 3m17.671s | 14,319.53 MB  | 28,342.81 MB  | 4 MB        | catabra  |

c) 500 Patients

Scaling further to 500 patients, MEDS-Tab maintained consistent performance, reinforcing its capability to manage large datasets efficiently. Because of the set time limit of ???, we could not get results for Catabra and TSFresh.

**Table 5: 48,115 Codes, 795,368 Rows, Output Shape \[75,595, 8,115\]**

| Wall Time | Avg Memory  | Peak Memory | Output Size | Method   |
| --------- | ----------- | ----------- | ----------- | -------- |
| 0m15.867s | 1,410.79 MB | 3,539.32 MB | 442 MB      | meds_tab |

______________________________________________________________________

# XGBoost Performance

## D.1 XGBoost Model Performance on MIMIC-IV Across Tasks

Evaluating our tabularization approach for baseline models involved training XGBoost across a spectrum of binary clinical prediction tasks using data from the MIMIC-IV database. These tasks encompassed diverse outcomes such as mortality predictions over different intervals, readmission predictions, and lengths of stay (LOS) in both ICU and hospital settings.

Each task is characterized by its specific label and prediction time. For instance, predicting "30-day readmission" involves assessing whether a patient returns to the hospital within 30 days, with predictions made at the time of discharge. This allows input features to be derived from the entire duration of the patient's admission. In contrast, tasks like "In ICU Mortality" focus on predicting the occurrence of death using only data from the first 24 or 48 hours of ICU admission. Specifically, we use the terminology "Index Timestamp" to mean the timestamp such that no event included as input will occur later than this point.

Optimizing predictive accuracy and model performance necessitated employing varied window sizes and aggregations of patient data. This approach allows us to effectively capture and leverage the temporal dynamics and clinical nuances inherent in each prediction task.

### 1.1 XGBoost Time and Memory Profiling on MIMIC-IV

A single XGBoost run was completed to profile time and memory usage. This was done for each `$TASK` using the following command:

```
meds-tab-xgboost
      MEDS_cohort_dir="path_to_data" \
      task_name=$TASK \
      output_dir="output_directory" \
      do_overwrite=False \
```

This uses the defaults minimum code inclusion frequency, window sizes, and aggregations from the `launch_xgboost.yaml`:

```yaml
allowed_codes:      # allows all codes that meet min code inclusion frequency
min_code_inclusion_frequency: 10
window_sizes:
  - 1d
  - 7d
  - 30d
  - 365d
  - full
aggs:
  - static/present
  - static/first
  - code/count
  - value/count
  - value/sum
  - value/sum_sqd
  - value/min
  - value/max
```

Since this includes every window size and aggregation, it is the most expoensive to run. The runtimes and memory usage are reported below.

#### 1.1.1 XGBoost Runtimes and Memory Usage on MIMIC-IV Tasks

| Task                            | Index Timestamp   | Real Time | User Time | Sys Time  | Avg Memory (MiB) | Peak Memory (MiB) |
| ------------------------------- | ----------------- | --------- | --------- | --------- | ---------------- | ----------------- |
| Post-discharge 30 day Mortality | Discharge         | 2m59.807s | 3m38.285s | 0m38.274s | 9036.728         | 11955.312         |
| Post-discharge 1 year Mortality | Discharge         | 5m16.958s | 6m10.065s | 0m58.964s | 10803.670        | 12330.355         |
| 30 day Readmission              | Discharge         | 2m30.609s | 3m3.836s  | 0m39.415s | 13198.832        | 18676.605         |
| In ICU Mortality                | Admission + 24 hr | 0m38.275s | 1m3.656s  | 0m13.198s | 1711.942         | 2985.699          |
| In ICU Mortality                | Admission + 48 hr | 0m34.669s | 1m1.389s  | 0m12.742s | 1613.256         | 2770.438          |
| In Hospital Mortality           | Admission + 24 hr | 2m8.912s  | 2m41.180s | 0m31.707s | 9071.615         | 12055.980         |
| In Hospital Mortality           | Admission + 48 hr | 1m54.025s | 2m25.322s | 0m28.925s | 8857.783         | 12370.898         |
| LOS in ICU > 3 days             | Admission + 24 hr | 2m2.689s  | 2m37.196s | 0m28.110s | 4650.008         | 5715.484          |
| LOS in ICU > 3 days             | Admission + 48 hr | 1m44.434s | 2m18.558s | 0m23.825s | 4453.363         | 5577.164          |
| LOS in Hospital > 3 days        | Admission + 24 hr | 6m4.884s  | 7m5.025s  | 1m4.335s  | 11011.710        | 12223.449         |
| LOS in Hospital > 3 days        | Admission + 48 hr | 6m9.587s  | 7m12.853s | 1m3.858s  | 10703.064        | 11829.742         |

#### 1.1.2 MIMIC-IV Task Specific Training Cohort Size

To better understand the runtimes, we also report the task specific cohort size.

| Task                            | Index Timestamp   | Number of Patients | Number of Events |
| ------------------------------- | ----------------- | ------------------ | ---------------- |
| Post-discharge 30 day Mortality | Discharge         | 149014             | 356398           |
| Post-discharge 1 year Mortality | Discharge         | 149014             | 356398           |
| 30 day Readmission              | Discharge         | 17418              | 377785           |
| In ICU Mortality                | Admission + 24 hr | 7839               | 22811            |
| In ICU Mortality                | Admission + 48 hr | 6750               | 20802            |
| In Hospital Mortality           | Admission + 24 hr | 51340              | 338614           |
| In Hospital Mortality           | Admission + 48 hr | 47231              | 348289           |
| LOS in ICU > 3 days             | Admission + 24 hr | 42809              | 61342            |
| LOS in ICU > 3 days             | Admission + 48 hr | 42805              | 61327            |
| LOS in Hospital > 3 days        | Admission + 24 hr | 152126             | 360208           |
| LOS in Hospital > 3 days        | Admission + 48 hr | 152120             | 359020           |

### 1.2 MIMIC-IV Sweep

The XGBoost sweep was run using the following command for each `$TASK`:

```
meds-tab-xgboost --multirun \
      MEDS_cohort_dir="path_to_data" \
      task_name=$TASK \
      output_dir="output_directory" \
      tabularization.window_sizes=$(generate-permutations [1d,30d,365d,full]) \
      do_overwrite=False \
      tabularization.aggs=$(generate-permutations [static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max])
```

The model parameters were set to:

```yaml
model:
  booster: gbtree
  device: cpu
  nthread: 1
  tree_method: hist
  objective: binary:logistic
```

The hydra sweeper swept over the parameters:

```yaml
params:
  +model_params.model.eta: tag(log, interval(0.001, 1))
  +model_params.model.lambda: tag(log, interval(0.001, 1))
  +model_params.model.alpha: tag(log, interval(0.001, 1))
  +model_params.model.subsample: interval(0.5, 1)
  +model_params.model.min_child_weight: interval(1e-2, 100)
  +model_params.model.max_depth: range(2, 16)
  model_params.num_boost_round: range(100, 1000)
  model_params.early_stopping_rounds: range(1, 10)
  tabularization.min_code_inclusion_frequency: tag(log, range(10, 1000000))
```

Note that the XGBoost command shown includes `tabularization.window_sizes` and ` tabularization.aggs` in the parameters to sweep over.

For a complete example on MIMIC-IV and for all of our config files, see the [MIMIC-IV companion repository](https://github.com/mmcdermott/MEDS_TAB_MIMIC_IV).

#### 1.2.1 XGBoost Performance on MIMIC-IV

| Task                            | Index Timestamp   | AUC          | Minimum Code Inclusion Frequency | Number of Included Codes\* | Window Sizes           | Aggregations                                                                |
| ------------------------------- | ----------------- | ------------ | -------------------------------- | -------------------------- | ---------------------- | --------------------------------------------------------------------------- |
| Post-discharge 30 day Mortality | Discharge         | 0.9347633541 | 1371                             | 5712                       | \[7d,full\]            | \[code/count,value/count,value/min,value/max\]                              |
| Post-discharge 1 year Mortality | Discharge         | 0.8979986449 | 289                              | 10048                      | \[2h,12h,1d,30d,full\] | \[static/present,code/count,value/sum_sqd,value/min\]                       |
| 30 day Readmission              | Discharge         | 0.7076685566 | 303                              | 9903                       | \[30d,365d,full\]      | \[code/count,value/count,value/sum,value/sum_sqd,value/max\]                |
| In ICU Mortality                | Admission + 24 hr | 0.6612338171 | 7059                             | 3037                       | \[12h,full\]           | \[static/present,code/count,value/sum,value/min,value/max\]                 |
| In ICU Mortality                | Admission + 48 hr | 0.671986067  | 71                               | 16112                      | \[1d,7d,full\]         | \[static/present,code/count,value/sum,value/min,value/max\]                 |
| In Hospital Mortality           | Admission + 24 hr | 0.8119187166 | 43                               | 18989                      | \[1d,full\]            | \[static/present,code/count,value/sum,value/min,value/max\]                 |
| In Hospital Mortality           | Admission + 48 hr | 0.8100362943 | 678                              | 7433                       | \[1d,full\]            | \[static/present,code/count,value/count\]                                   |
| LOS in ICU > 3 days             | Admission + 24 hr | 0.9455099633 | 30443                            | 1624                       | \[2h,7d,30d\]          | \[static/present,code/count,value/count,value/sum,value/sum_sqd,value/max\] |
| LOS in ICU > 3 days             | Admission + 48 hr | 0.9667108784 | 2864                             | 4332                       | \[2h,7d,30d\]          | \[code/count,value/sum_sqd,value/max\]                                      |
| LOS in Hospital > 3 days        | Admission + 24 hr | 0.9434966078 | 94633                            | 912                        | \[12h,1d,7d\]          | \[code/count,value/count,value/sum_sqd\]                                    |
| LOS in Hospital > 3 days        | Admission + 48 hr | 0.9449770561 | 30880                            | 1619                       | \[1d,7d,30d\]          | \[code/count,value/sum,value/min,value/max\]                                |

- Number of Included Codes is based on Minimum Code Inclusion Frequency -- we calculated the number of resulting codes that were above the minimum threshold and reported that.

#### 1.2.2 XGBoost Optimal Found Model Parameters

Additionally, the model parameters from the highest performing run are reported below.

| Task                            | Index Timestamp   | Eta            | Lambda         | Alpha         | Subsample    | Minimum Child Weight | Number of Boosting Rounds | Early Stopping Rounds | Max Tree Depth |
| ------------------------------- | ----------------- | -------------- | -------------- | ------------- | ------------ | -------------------- | ------------------------- | --------------------- | -------------- |
| Post-discharge 30 day Mortality | Discharge         | 0.005630897092 | 0.03218837176  | 0.3741846464  | 0.5716492359 | 52.66844896          | 703                       | 9                     | 16             |
| Post-discharge 1 year Mortality | Discharge         | 0.008978198787 | 0.086075240914 | 0.342564218   | 0.8994363088 | 75.94359197          | 858                       | 9                     | 11             |
| 30 day Readmission              | Discharge         | 0.005970244514 | 0.3591376982   | 0.3741846464  | 0.673450045  | 52.66844896          | 712                       | 9                     | 16             |
| In ICU Mortality                | Admission + 24 hr | 0.03824348927  | 0.06183970736  | 0.2310791064  | 0.9947482627 | 88.53086045          | 513                       | 7                     | 14             |
| In ICU Mortality (first 48h)    | Admission + 48 hr | 0.04373178504  | 0.04100575186  | 0.2888938852  | 0.9617417624 | 90.881739            | 484                       | 5                     | 14             |
| In Hospital Mortality           | Admission + 24 hr | 0.02790651024  | 0.01319397229  | 0.0105408763  | 0.5673852112 | 11.22281297          | 454                       | 6                     | 9              |
| In Hospital Mortality           | Admission + 48 hr | 0.01076063059  | 0.06007544254  | 0.1791900222  | 0.9641152835 | 83.69584368          | 631                       | 7                     | 13             |
| LOS in ICU > 3 days             | Admission + 24 hr | 0.01203878234  | 0.08963582145  | 0.1367180869  | 0.6264012852 | 26.20493325          | 650                       | 8                     | 14             |
| LOS in ICU > 3 days             | Admission + 48 hr | 0.01203878234  | 0.04882102808  | 0.1997059646  | 0.9608288859 | 83.9736355           | 615                       | 7                     | 13             |
| LOS in Hospital > 3 days        | Admission + 24 hr | 0.008389745342 | 0.06656965098  | 0.2553069741  | 0.9886841026 | 89.89987526          | 526                       | 5                     | 14             |
| LOS in Hospital > 3 days        | Admission + 48 hr | 0.00121145622  | 0.03018152667  | 0.02812771908 | 0.9671829656 | 8.657613623          | 538                       | 8                     | 7              |
