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

______________________________________________________________________

# Usage

This repository consists of two key pieces:

1. Construction and efficient loading of tabular (flat, non-longitudinal) summary features describing patient records in MEDS over arbitrary time windows (e.g. 1 year, 6 months, etc.), which go backward in time from a given index date.
2. Running a basic XGBoost AutoML pipeline over these tabular features to predict arbitrary binary classification or regression downstream tasks defined over these datasets. The "AutoML" part of this is not particularly advanced -- what is more advanced is the efficient construction, storage, and loading of tabular features for the candidate AutoML models, enabling a far more extensive search over a much larger total number of features than prior systems.

## Quick Start

To use MEDS-Tab, install the dependencies following commands below. Note that this version of MEDS-Tab is
compatible with [MEDS v0.3](https://github.com/Medical-Event-Data-Standard/meds/releases/tag/0.3.0)

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

For an end to end example, including re-sharding the input via MEDS-Transforms, see
[this example script](https://gist.github.com/mmcdermott/34194e484d7b2a2f68967b9bbccfb35b)

See [`/tests/test_integration.py`](https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/main/tests/test_integration.py) for a local example of the end-to-end pipeline (minus re-sharding) being run on synthetic data. This script is a functional test that is also run with `pytest` to verify the correctness of the algorithm.

## Why MEDS-Tab?

MEDS-Tab is a comprehensive framework designed to streamline the handling, modeling, and analysis of complex medical time-series data. By leveraging automated processes, MEDS-Tab significantly reduces the computation required to generate high-quality baseline models for diverse supervised learning tasks.

- Cost Efficiency: MEDS-Tab is dramatically more cost-effective compared to existing solutions
- Strong Performance: MEDS-Tab provides robustness and high performance across various datasets compared with other frameworks.

### I. Transform to MEDS

MEDS-Tab leverages the recently developed, minimal, easy-to-use Medical Event Data Standard (MEDS) schema to standardize structured EHR data to a consistent schema from which baselines can be reliably produced across arbitrary tasks and settings. In order to use MEDS-Tab, you will first need to transform your raw EHR data to a MEDS format, which can be done using the following libraries:

- [MEDS Polars](https://github.com/mmcdermott/MEDS_polars_functions) for a set of functions and scripts for extraction to and transformation/pre-processing of MEDS-formatted data.
- [MEDS ETL](https://github.com/Medical-Event-Data-Standard/meds_etl) for a collection of ETLs from common data formats to MEDS. The package library currently supports MIMIC-IV, OMOP v5, and MEDS FLAT (a flat version of MEDS).

### II. Run MEDS-Tab

- Run the MEDS-Tab Command-Line Interface tool (`MEDS-Tab-cli`) to extract cohorts based on your task - check out the [Usage Guide](https://meds-tab--36.org.readthedocs.build/en/36/overview.html#core-cli-scripts-overview)!

- Painless Reproducibility: Use [MEDS-Tab](https://github.com/mmcdermott/MEDS_TAB_MIMIC_IV/tree/main/tasks) to obtain comparable, reproducible, and well-tuned XGBoost results tailored to your dataset-specific feature space!

By following these steps, you can seamlessly transform your dataset, define necessary criteria, and leverage powerful machine learning tools within the MEDS-Tab ecosystem. This approach not only simplifies the process but also ensures high-quality, reproducible results for your machine learning tasks for health projects. It can reliably take no more than a week of full-time human effort to perform Steps I-V on new datasets in reasonable raw formulations!

## Core CLI Scripts Overview

0. First, if your data is not already sharded to the degree you want and in a manner that subdivides your
   splits with the format `"$SPLIT_NAME/\d+.parquet"`, where `$SPLIT_NAME` does not contain slashes, you will
   need to re-shard your data. This can be done via the
   [MEDS-Transforms](https://github.com/mmcdermott/MEDS_transforms) library, which is not included in this
   repository. Having data sharded by split _is a necessary step_ to ensure that the data is efficiently
   processed in parallel. You can easily re-shard your input MEDS cohort in the environment into which this
   package is installed with the following command:

   ```console
   # Re-shard pipeline
   # $MIMICIV_input_dir is the directory containing the input, MEDS v0.3 formatted MIMIC-IV data
   # $MEDS_TAB_COHORT_DIR is the directory where the re-sharded MEDS dataset will be stored, and where your model
   # will store cached files during processing by default.
   # $N_PATIENTS_PER_SHARD is the number of patients per shard you want to use.
   MEDS_transform-reshard_to_split \
       input_dir="$MIMICIV_input_dir" \
       cohort_dir="$MEDS_TAB_COHORT_DIR" \
       'stages=["reshard_to_split"]' \
       stage="reshard_to_split" \
       stage_configs.reshard_to_split.n_patients_per_shard=$N_PATIENTS_PER_SHARD
   ```

1. **`meds-tab-describe`**: This command processes MEDS data shards to compute the frequencies of different code types. It differentiates codes into the following categories:

   - time-series codes (codes with timestamps)
   - time-series numerical values (codes with timestamps and numerical values)
   - static codes (codes without timestamps)
   - static numerical codes (codes without timestamps but with numerical values).

   This script further caches feature names and frequencies in a dataset stored in a `code_metadata.parquet` file within the `input_dir` argument specified as a hydra-style command line argument.

2. **`meds-tab-tabularize-static`**: Filters and processes the dataset based on the frequency of codes, generating a tabular vector for each patient at each timestamp in the shards. Each row corresponds to a unique `subject_id` and `timestamp` combination, thus rows are duplicated across multiple timestamps for the same patient.

   **Example: Tabularizing static data** with the minimum code frequency of 10, window sizes of `[1d, 30d,  365d, full]`, and value aggregation methods of `[static/present, static/first, code/count, value/count, value/sum, value/sum_sqd, value/min, value/max]`

   ```console
   meds-tab-tabularize-static input_dir="path_to_data" \
                               tabularization.min_code_inclusion_frequency=10 \
                               tabularization.window_sizes=[1d,30d,365d,full] \
                               do_overwrite=False \
                               tabularization.aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]"
   ```

   - For the exhaustive examples of value aggregations, see [`/src/MEDS_tabular_automl/utils.py`](https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/main/src/MEDS_tabular_automl/utils.py#L24)

3. **`meds-tab-tabularize-time-series`**: Iterates through combinations of a shard, `window_size`, and `aggregation` to generate feature vectors that aggregate patient data for each unique `subject_id` x `timestamp`. This stage (and the previous stage) uses sparse matrix formats to efficiently handle the computational and storage demands of rolling window calculations on large datasets. We support parallelization through Hydra's [`--multirun`](https://hydra.cc/docs/intro/#multirun) flag and the [`joblib` launcher](https://hydra.cc/docs/plugins/joblib_launcher/#internaldocs-banner).

   **Example: Aggregate time-series data** on features across different `window_sizes`

   ```console
   meds-tab-tabularize-time-series --multirun \
      worker="range(0,$N_PARALLEL_WORKERS)" \
      hydra/launcher=joblib \
      input_dir="path_to_data" \
      tabularization.min_code_inclusion_frequency=10 \
      do_overwrite=False \
      tabularization.window_sizes=[1d,30d,365d,full] \
      tabularization.aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
   ```

4. **`meds-tab-cache-task`**: Aligns task-specific labels with the nearest prior event in the tabularized data. It requires a labeled dataset directory with three columns (`subject_id`, `timestamp`, `label`) structured similarly to the `input_dir`.

   **Example: Align tabularized data** for a specific task `$TASK` and labels that have been pulled from [ACES](https://github.com/justin13601/ACES)

   ```console
   meds-tab-cache-task input_dir="path_to_data" \
      task_name=$TASK \
      tabularization.min_code_inclusion_frequency=10 \
      do_overwrite=False \
      tabularization.window_sizes=[1d,30d,365d,full] \
      tabularization.aggs=[static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
   ```

5. **`meds-tab-xgboost`**: Trains an XGBoost model using user-specified parameters. Permutations of `window_sizes` and `aggs` can be generated using `generate-subsets` command (See the section below for descriptions).

   ```console
   meds-tab-xgboost --multirun \
      input_dir="path_to_data" \
      task_name=$TASK \
      output_dir="output_directory" \
      tabularization.min_code_inclusion_frequency=10 \
      tabularization.window_sizes=$(generate-subsets [1d,30d,365d,full]) \
      do_overwrite=False \
      tabularization.aggs=$(generate-subsets [static/present,static/first,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max])
   ```

## Additional CLI Scripts

1. **`generate-subsets`**: Generates and prints a sorted list of all non-empty subsets from a comma-separated input. This is provided for the convenience of sweeping over all possible combinations of window sizes and aggregations.

   For example, you can directly call **`generate-subsets`** in the command line:

   ```console
   generate-subsets [2,3,4] \
   [2], [2, 3], [2, 3, 4], [2, 4], [3], [3, 4], [4]
   ```

   This could be used in the command line in concert with other calls. For example, the following call:

   ```console
   meds-tab-xgboost --multirun tabularization.window_sizes=$(generate-subsets [1d,2d,7d,full])
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

______________________________________________________________________

# The MEDS-Tab Architecture

In this section, we describe the MEDS-Tab architecture, specifically some of the pipeline choices we made to reduce memory usage and increase speed during the tabularization process and XGBoost tuning process.

We break our method into 4 discrete parts:

1. Describe codes (compute feature frequencies)
2. Tabularization of time-series data
3. Efficient data caching for task-specific rows
4. XGBoost training

## 1. Describe Codes (compute feature frequencies)

This initial stage processes a pre-shareded dataset. We expect a structure as follows where each shard contains a subset of the patients:

```text
/PATH/TO/MEDS/DATA
│
└─── <SPLIT A>
│   │   <SHARD 0>.parquet
│   │   <SHARD 1>.parquet
│   │   ...
│
└─── <SPLIT B>
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

## 2. Tabularization of Time-Series Data

### Overview

The tabularization stage of our pipeline, exposed via the cli commands:

- `meds-tab-tabularize-static` for tabularizing static data
- and `meds-tab-tabularize-time-series` for tabularizing the time series data

Static data is relatively small in the medical datasets, so we use a dense pivot operation, convert it to a sparse matrix, and then duplicate rows such that the static data will match up with the time series data rows generated in the next step. Static data is currently processed serially.

The script for tabularizing time series data primarily transforms a raw, unstructured dataset into a structured, feature-rich dataset by utilizing a series of sophisticated data processing steps. This transformation (as depicted in the figure below) involves converting raw time series from a Polars dataframe into a sparse matrix format, aggregating events that occur at the same date for the same patient, and then applying rolling window aggregations to extract temporal features.

![Time Series Tabularization Method](docs/assets/pivot.png)

### High-Level Tabularization Algorithm

1. **Data Loading and Categorization**:

   - The script iterates through shards of patients, and shards can be processed in parallel using hydras joblib to launch multiple processes.

2. **Sparse Matrix Conversion**:

   - Data from the Polars dataframe is converted into a sparse matrix format, where each row represents a unique event (patient x timestamp), and each column corresponds to a MEDS code for the patient.

3. **Rolling Window Aggregation**:

   - For each aggregation method (sum, count, min, max, etc.), events that occur on the same date for the same patient are aggregated. This reduces the amount of data we have to perform rolling windows over.
   - Then we aggregate features over the specified rolling windows sizes.

4. **Output Storage**:

   - Sparse array is converted to Coordinate List format and stored as a `.npz` file on disk.
   - The file paths look as follows

```text
/PATH/TO/MEDS/TABULAR_DATA
│
└─── <SPLIT A>
    ├─── <SHARD 0>
    │   ├───code
    │   │   └───count.npz
    │   └───value
    │       └───sum.npz
    ...
```

## 3. Efficient Data Caching for Task-Specific Rows

Now that we have generated tabular features for all the events in our dataset, we can cache subsets relevant for each task we wish to train a supervised model on. This step is critical for efficiently training machine learning models on task-specific data without having to load the entire dataset.

**Detailed Workflow:**

- **Row Selection Based on Tasks**: Only the data rows that are relevant to the specific tasks are selected and cached. This reduces the memory footprint and speeds up the training process.
- **Use of Sparse Matrices for Efficient Storage**: Sparse matrices are again employed here to store the selected data efficiently, ensuring that only non-zero data points are kept in memory, thus optimizing both storage and retrieval times.

The file structure for the cached data mirrors that of the tabular data, also consisting of `.npz` files, where users must specify the directory that stores labels. Labels follow the same shard file structure as the input meds data from step (1), and the label parquets need `subject_id`, `timestamp`, and `label` columns.

## 4. XGBoost Training

The final stage uses the processed and cached data to train an XGBoost model. This stage is optimized to handle the sparse data structures produced in earlier stages efficiently.

**Detailed Workflow:**

- **Iterator for Data Loading**: Custom iterators are designed to load sparse matrices efficiently into the XGBoost training process, which can handle sparse inputs natively, thus maintaining high computational efficiency.
- **Training and Validation**: The model is trained using the tabular data, with evaluation steps that include early stopping to prevent overfitting and tuning of hyperparameters based on validation performance.
- **Hyperparameter Tuning**: We use [optuna](https://optuna.org/) to tune over XGBoost model parameters, aggregations, window sizes, and the minimum code inclusion frequency.

______________________________________________________________________

# Computational Performance vs. Existing Pipelines

Evaluating the computational overhead of tabularization methods is essential for assessing their efficiency and suitability for large-scale medical data processing. This section presents a comparative analysis of the computational overhead of MEDS-Tab with other systems like Catabra and TSFresh. It outlines the performance of each system in terms of wall time, memory usage, and output size, highlighting the computational efficiency and scalability of MEDS-Tab.

## 1. System Comparison Overview

The systems compared in this study represent different approaches to data tabularization, with the main difference being MEDS-Tab usage of sparse tabularization. Specifically, for comparison we used:

1. **Catabra/Catabra-Mem**: Offers data processing capabilities for time-series medical data, with variations to test memory management.
2. **TSFresh**: Both known and used for extensive feature extraction capabilities.

The benchmarking tests were conducted using the following hardware and software settings:

- **CPU Specification**: 2 x AMD EPYC 7713 64-Core Processor
- **RAM Specification**: 1024GB, 3200MHz, DDR4
- **Software Environment**: Ubuntu 22.04.4 LTS

### MEDS-Tab Tabularization Technique

Tabularization of time-series data, as depicted above, is commonly used in several past works. The only two libraries to our knowledge that provide a full tabularization pipeline are `tsfresh` and `catabra`. `catabra` also offers a slower but more memory-efficient version of their method which we denote `catabra-mem`. Other libraries either provide only rolling window functionalities (`featuretools`) or just pivoting operations (`Temporai`/`Clairvoyance`, `sktime`, `AutoTS`). We provide a significantly faster and more memory-efficient method. Our findings show that on the MIMIC-IV and eICU medical datasets, we significantly outperform both above-mentioned methods that provide similar functionalities with MEDS-Tab. While `catabra` and `tsfresh` could not even run within a budget of 10 minutes on as low as 10 patients' data for eICU, our method scales to process hundreds of patients with low memory usage under the same time budget. We present the results below.

## 2. Comparative Performance Analysis

The tables below detail computational resource utilization across two datasets and various patient scales, emphasizing the better performance of MEDS-Tab in all of the scenarios. The tables are organized by dataset and number of patients. For the analysis, the full window sizes and the aggregation method code_count were used. Additionally, we use a budget of 10 minutes for running our tests given that for such a small number of patients (10, 100, and 500 patients) data should be processed quickly. Note that `catabra-mem` is omitted from the tables as it was never completed within the 10-minute budget.

### eICU Dataset

The only method that was able to tabularize eICU data was MEDS-Tab. We ran our method with both 100 and 500 patients, resulting in an increment of three times in the number of codes. MEDS-Tab gave efficient results in terms of both time and memory usage.

a) 100 Patients

**Table 1: 6,374 Codes, 2,065,608 Rows, Output Shape \[132,461, 6,374\]**

| Wall Time | Avg Memory | Peak Memory | Output Size | Method   |
| --------- | ---------- | ----------- | ----------- | -------- |
| 0m39s     | 5,271 MB   | 14,791 MB   | 362 MB      | meds_tab |

b) 500 Patients

**Table 2: 18,314 Codes, 8,737,355 Rows, Output Shape \[565,014, 18,314\]**

| Wall Time | Avg Memory | Peak Memory | Output Size | Method   |
| --------- | ---------- | ----------- | ----------- | -------- |
| 3m4s      | 8,335 MB   | 15,102 MB   | 1,326 MB    | meds_tab |

### MIMIC-IV Dataset

MEDS-Tab, `tsfresh`, and `catabra` were tested across three different patient scales on MIMIC-IV.

a) 10 Patients

This table illustrates the efficiency of MEDS-Tab in processing a small subset of patients with extremely low computational cost and high data throughput, outperforming `tsfresh` and `catabra` in terms of both time and memory efficiency.

**Table 3: 1,504 Codes, 23,346 Rows, Output Shape \[2,127, 1,504\]**

| Wall Time | Avg Memory | Peak Memory | Output Size | Method   |
| --------- | ---------- | ----------- | ----------- | -------- |
| 0m2s      | 423 MB     | 943 MB      | 7 MB        | meds_tab |
| 1m41s     | 84,159 MB  | 265,877 MB  | 1 MB        | tsfresh  |
| 0m15s     | 2,537 MB   | 4,781 MB    | 1 MB        | catabra  |

b) 100 Patients

The performance gap was further highlighted with an increased number of patients and codes. For a moderate patient count, MEDS-Tab demonstrated superior performance with significantly lower wall times and memory usage compared to `tsfresh` and `catabra`.

**Table 4: 4,154 Codes, 150,789 Rows, Output Shape \[15,664, 4,154\]**

| Wall Time | Avg Memory | Peak Memory | Output Size | Method   |
| --------- | ---------- | ----------- | ----------- | -------- |
| 0m5s      | 718 MB     | 1,167 MB    | 45 MB       | meds_tab |
| 5m9s      | 217,477 MB | 659,735 MB  | 4 MB        | tsfresh  |
| 3m17s     | 14,319 MB  | 28,342 MB   | 4 MB        | catabra  |

c) 500 Patients

Scaling further to 500 patients, MEDS-Tab maintained consistent performance, reinforcing its capability to manage large datasets efficiently. Because of the set time limit of 10 minutes, we could not get results for `catabra` and `tsfresh`. In comparison, MEDS-Tab processed the data in about 15 seconds, making it at least 40 times faster for the given patient scale.

**Table 5: 48,115 Codes, 795,368 Rows, Output Shape \[75,595, 8,115\]**

| Wall Time | Avg Memory | Peak Memory | Output Size | Method   |
| --------- | ---------- | ----------- | ----------- | -------- |
| 0m16s     | 1,410 MB   | 3,539 MB    | 442 MB      | meds_tab |

______________________________________________________________________

# Prediction Performance

## XGBoost Model Performance on MIMIC-IV Tasks

Evaluating our tabularization approach for baseline models involved training XGBoost across a spectrum of binary clinical prediction tasks, using data from the MIMIC-IV database. These tasks encompassed diverse outcomes such as mortality predictions over different intervals, readmission predictions, and lengths of stay (LOS) in both ICU and hospital settings.

Each task is characterized by its specific label and prediction time. For instance, predicting "30-day readmission" involves assessing whether a patient returns to the hospital within 30 days, with predictions made at the time of discharge. This allows input features to be derived from the entire duration of the patient's admission. In contrast, tasks like "In ICU Mortality" focus on predicting the occurrence of death using only data from the first 24 or 48 hours of ICU admission. Specifically, we use the terminology "Index Timestamp" to mean the timestamp such that no event included as input will occur later than this point.

We optimize predictive accuracy and model performance by using varied window sizes and aggregations of patient data. This approach allows us to effectively capture and leverage the temporal dynamics and clinical nuances inherent in each prediction task.

### 1. XGBoost Time and Memory Profiling on MIMIC-IV

A single XGBoost run was completed to profile time and memory usage. This was done for each `$TASK` using the following command:

```console
meds-tab-xgboost
      input_dir="path_to_data" \
      task_name=$TASK \
      output_dir="output_directory" \
      do_overwrite=False \
```

This uses the default minimum code inclusion frequency, window sizes, and aggregations from the `launch_xgboost.yaml`:

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

Since this includes every window size and aggregation, it is the most expensive to run. The runtimes and memory usage are reported below.

#### 1.1 XGBoost Runtimes and Memory Usage on MIMIC-IV Tasks

| Task                            | Index Timestamp   | Real Time | User Time | Sys Time | Avg Memory (MiB) | Peak Memory (MiB) |
| ------------------------------- | ----------------- | --------- | --------- | -------- | ---------------- | ----------------- |
| Post-discharge 30 day Mortality | Discharge         | 2m59s     | 3m38s     | 0m38s    | 9,037            | 11,955            |
| Post-discharge 1 year Mortality | Discharge         | 5m16s     | 6m10s     | 0m59s    | 10,804           | 12,330            |
| 30 day Readmission              | Discharge         | 2m30s     | 3m3s      | 0m39s    | 13,199           | 18,677            |
| In ICU Mortality                | Admission + 24 hr | 0m38s     | 1m3s      | 0m13s    | 1,712            | 2,986             |
| In ICU Mortality                | Admission + 48 hr | 0m34s     | 1m1s      | 0m13s    | 1,613            | 2,770             |
| In Hospital Mortality           | Admission + 24 hr | 2m8s      | 2m41s     | 0m32s    | 9,072            | 12,056            |
| In Hospital Mortality           | Admission + 48 hr | 1m54s     | 2m25s     | 0m29s    | 8,858            | 12,371            |
| LOS in ICU > 3 days             | Admission + 24 hr | 2m3s      | 2m37s     | 0m28s    | 4,650            | 5,715             |
| LOS in ICU > 3 days             | Admission + 48 hr | 1m44s     | 2m18s     | 0m24s    | 4,453            | 5,577             |
| LOS in Hospital > 3 days        | Admission + 24 hr | 6m5s      | 7m5s      | 1m4s     | 11,012           | 12,223            |
| LOS in Hospital > 3 days        | Admission + 48 hr | 6m10s     | 7m12s     | 1m4s     | 10,703           | 11,830            |

#### 1.2 MIMIC-IV Task Specific Training Cohort Size

To better understand the runtimes, we also report the task specific cohort size.

| Task                            | Index Timestamp   | Number of Patients | Number of Events |
| ------------------------------- | ----------------- | ------------------ | ---------------- |
| Post-discharge 30 day Mortality | Discharge         | 149,014            | 356,398          |
| Post-discharge 1 year Mortality | Discharge         | 149,014            | 356,398          |
| 30 day Readmission              | Discharge         | 17,418             | 377,785          |
| In ICU Mortality                | Admission + 24 hr | 7,839              | 22,811           |
| In ICU Mortality                | Admission + 48 hr | 6,750              | 20,802           |
| In Hospital Mortality           | Admission + 24 hr | 51,340             | 338,614          |
| In Hospital Mortality           | Admission + 48 hr | 47,231             | 348,289          |
| LOS in ICU > 3 days             | Admission + 24 hr | 42,809             | 61,342           |
| LOS in ICU > 3 days             | Admission + 48 hr | 42,805             | 61,327           |
| LOS in Hospital > 3 days        | Admission + 24 hr | 152,126            | 360,208          |
| LOS in Hospital > 3 days        | Admission + 48 hr | 152,120            | 359,020          |

### 2. MIMIC-IV Sweep

The XGBoost sweep was run using the following command for each `$TASK`:

```console
meds-tab-xgboost --multirun \
      input_dir="path_to_data" \
      task_name=$TASK \
      output_dir="output_directory" \
      tabularization.window_sizes=$(generate-subsets [1d,30d,365d,full]) \
      do_overwrite=False \
      tabularization.aggs=$(generate-subsets [static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max])
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
  model.eta: tag(log, interval(0.001, 1))
  model.lambda: tag(log, interval(0.001, 1))
  model.alpha: tag(log, interval(0.001, 1))
  model.subsample: interval(0.5, 1)
  model.min_child_weight: interval(1e-2, 100)
  model.max_depth: range(2, 16)
  num_boost_round: range(100, 1000)
  early_stopping_rounds: range(1, 10)
  tabularization.min_code_inclusion_frequency: tag(log, range(10, 1000000))
```

Note that the XGBoost command shown includes `tabularization.window_sizes` and ` tabularization.aggs` in the parameters to sweep over.

For a complete example on MIMIC-IV and for all of our config files, see the [MIMIC-IV companion repository](https://github.com/mmcdermott/MEDS_TAB_MIMIC_IV).

#### 2.1 XGBoost Performance on MIMIC-IV

| Task                            | Index Timestamp   | AUC   | Minimum Code Inclusion Frequency | Number of Included Codes\* | Window Sizes           | Aggregations                                                                |
| ------------------------------- | ----------------- | ----- | -------------------------------- | -------------------------- | ---------------------- | --------------------------------------------------------------------------- |
| Post-discharge 30 day Mortality | Discharge         | 0.935 | 1,371                            | 5,712                      | \[7d,full\]            | \[code/count,value/count,value/min,value/max\]                              |
| Post-discharge 1 year Mortality | Discharge         | 0.898 | 289                              | 10,048                     | \[2h,12h,1d,30d,full\] | \[static/present,code/count,value/sum_sqd,value/min\]                       |
| 30 day Readmission              | Discharge         | 0.708 | 303                              | 9,903                      | \[30d,365d,full\]      | \[code/count,value/count,value/sum,value/sum_sqd,value/max\]                |
| In ICU Mortality                | Admission + 24 hr | 0.661 | 7,059                            | 3,037                      | \[12h,full\]           | \[static/present,code/count,value/sum,value/min,value/max\]                 |
| In ICU Mortality                | Admission + 48 hr | 0.673 | 71                               | 16,112                     | \[1d,7d,full\]         | \[static/present,code/count,value/sum,value/min,value/max\]                 |
| In Hospital Mortality           | Admission + 24 hr | 0.812 | 43                               | 18,989                     | \[1d,full\]            | \[static/present,code/count,value/sum,value/min,value/max\]                 |
| In Hospital Mortality           | Admission + 48 hr | 0.810 | 678                              | 7,433                      | \[1d,full\]            | \[static/present,code/count,value/count\]                                   |
| LOS in ICU > 3 days             | Admission + 24 hr | 0.946 | 30,443                           | 1,624                      | \[2h,7d,30d\]          | \[static/present,code/count,value/count,value/sum,value/sum_sqd,value/max\] |
| LOS in ICU > 3 days             | Admission + 48 hr | 0.967 | 2,864                            | 4,332                      | \[2h,7d,30d\]          | \[code/count,value/sum_sqd,value/max\]                                      |
| LOS in Hospital > 3 days        | Admission + 24 hr | 0.943 | 94,633                           | 912                        | \[12h,1d,7d\]          | \[code/count,value/count,value/sum_sqd\]                                    |
| LOS in Hospital > 3 days        | Admission + 48 hr | 0.945 | 30,880                           | 1,619                      | \[1d,7d,30d\]          | \[code/count,value/sum,value/min,value/max\]                                |

- Number of Included Codes is based on Minimum Code Inclusion Frequency -- we calculated the number of resulting codes that were above the minimum threshold and reported that.

#### 2.2 XGBoost Optimal Found Model Parameters

Additionally, the model parameters from the highest-performing run are reported below.

| Task                            | Index Timestamp   | Eta   | Lambda | Alpha | Subsample | Minimum Child Weight | Number of Boosting Rounds | Early Stopping Rounds | Max Tree Depth |
| ------------------------------- | ----------------- | ----- | ------ | ----- | --------- | -------------------- | ------------------------- | --------------------- | -------------- |
| Post-discharge 30 day Mortality | Discharge         | 0.006 | 0.032  | 0.374 | 0.572     | 53                   | 703                       | 9                     | 16             |
| Post-discharge 1 year Mortality | Discharge         | 0.009 | 0.086  | 0.343 | 0.899     | 76                   | 858                       | 9                     | 11             |
| 30 day Readmission              | Discharge         | 0.006 | 0.359  | 0.374 | 0.673     | 53                   | 712                       | 9                     | 16             |
| In ICU Mortality                | Admission + 24 hr | 0.038 | 0.062  | 0.231 | 0.995     | 89                   | 513                       | 7                     | 14             |
| In ICU Mortality (first 48h)    | Admission + 48 hr | 0.044 | 0.041  | 0.289 | 0.961     | 91                   | 484                       | 5                     | 14             |
| In Hospital Mortality           | Admission + 24 hr | 0.028 | 0.013  | 0.011 | 0.567     | 11                   | 454                       | 6                     | 9              |
| In Hospital Mortality           | Admission + 48 hr | 0.011 | 0.060  | 0.179 | 0.964     | 84                   | 631                       | 7                     | 13             |
| LOS in ICU > 3 days             | Admission + 24 hr | 0.012 | 0.090  | 0.137 | 0.626     | 26                   | 650                       | 8                     | 14             |
| LOS in ICU > 3 days             | Admission + 48 hr | 0.012 | 0.049  | 0.200 | 0.960     | 84                   | 615                       | 7                     | 13             |
| LOS in Hospital > 3 days        | Admission + 24 hr | 0.008 | 0.067  | 0.255 | 0.989     | 90                   | 526                       | 5                     | 14             |
| LOS in Hospital > 3 days        | Admission + 48 hr | 0.001 | 0.030  | 0.028 | 0.967     | 9                    | 538                       | 8                     | 7              |

## XGBoost Model Performance on eICU Tasks

### eICU Sweep

The eICU sweep was conducted equivalently to the MIMIC-IV sweep. Please refer to the MIMIC-IV Sweep subsection above for details on the commands and sweep parameters.

For more details about eICU-specific task generation and running, see the [eICU companion repository](https://github.com/mmcdermott/MEDS_TAB_EICU).

#### 1. XGBoost Performance on eICU

| Task                            | Index Timestamp   | AUC   | Minimum Code Inclusion Frequency | Window Sizes             | Aggregations                                                   |
| ------------------------------- | ----------------- | ----- | -------------------------------- | ------------------------ | -------------------------------------------------------------- |
| Post-discharge 30 day Mortality | Discharge         | 0.603 | 68,235                           | \[12h,1d,full\]          | \[code/count,value/sum_sqd,value/max\]                         |
| Post-discharge 1 year Mortality | Discharge         | 0.875 | 3,280                            | \[30d,365d\]             | \[static/present,value/sum,value/sum_sqd,value/min,value/max\] |
| In Hospital Mortality           | Admission + 24 hr | 0.855 | 335,912                          | \[2h,7d,30d,365d,full\]  | \[static/present,code/count,value/count,value/min,value/max\]  |
| In Hospital Mortality           | Admission + 48 hr | 0.570 | 89,121                           | \[12h,1d,30d\]           | \[code/count,value/count,value/min\]                           |
| LOS in ICU > 3 days             | Admission + 24 hr | 0.783 | 7,881                            | \[1d,30d,full\]          | \[static/present,code/count,value/count,value/sum,value/max\]  |
| LOS in ICU > 3 days             | Admission + 48 hr | 0.757 | 1,719                            | \[2h,12h,7d,30d,full\]   | \[code/count,value/count,value/sum,value/sum_sqd,value/min\]   |
| LOS in Hospital > 3 days        | Admission + 24 hr | 0.864 | 160                              | \[1d,30d,365d,full\]     | \[static/present,code/count,value/min,value/max\]              |
| LOS in Hospital > 3 days        | Admission + 48 hr | 0.895 | 975                              | \[12h,1d,30d,365d,full\] | \[code/count,value/count,value/sum,value/sum_sqd\]             |

#### 2. XGBoost Optimal Found Model Parameters

| Task                            | Index Timestamp   | Eta   | Lambda | Alpha | Subsample | Minimum Child Weight | Number of Boosting Rounds | Early Stopping Rounds | Max Tree Depth |
| ------------------------------- | ----------------- | ----- | ------ | ----- | --------- | -------------------- | ------------------------- | --------------------- | -------------- |
| In Hospital Mortality           | Admission + 24 hr | 0.043 | 0.001  | 0.343 | 0.879     | 13                   | 574                       | 9                     | 14             |
| In Hospital Mortality           | Admission + 48 hr | 0.002 | 0.002  | 0.303 | 0.725     | 0                    | 939                       | 9                     | 12             |
| LOS in ICU > 3 days             | Admission + 24 hr | 0.210 | 0.189  | 0.053 | 0.955     | 5                    | 359                       | 6                     | 14             |
| LOS in ICU > 3 days             | Admission + 48 hr | 0.340 | 0.393  | 0.004 | 0.900     | 6                    | 394                       | 10                    | 13             |
| LOS in Hospital > 3 days        | Admission + 24 hr | 0.026 | 0.238  | 0.033 | 0.940     | 46                   | 909                       | 5                     | 11             |
| LOS in Hospital > 3 days        | Admission + 48 hr | 0.100 | 0.590  | 0.015 | 0.914     | 58                   | 499                       | 10                    | 9              |
| Post-discharge 30 day Mortality | Discharge         | 0.003 | 0.0116 | 0.001 | 0.730     | 13                   | 986                       | 7                     | 7              |
| Post-discharge 1 year Mortality | Discharge         | 0.005 | 0.006  | 0.002 | 0.690     | 93                   | 938                       | 6                     | 14             |

#### 3. eICU Task Specific Training Cohort Size

| Task                            | Index Timestamp   | Number of Patients | Number of Events |
| ------------------------------- | ----------------- | ------------------ | ---------------- |
| Post-discharge 30 day Mortality | Discharge         | 91,405             | 91,405           |
| Post-discharge 1 year Mortality | Discharge         | 91,405             | 91,405           |
| In Hospital Mortality           | Admission + 24 hr | 35,85              | 3,585            |
| In Hospital Mortality           | Admission + 48 hr | 1,527              | 1,527            |
| LOS in ICU > 3 days             | Admission + 24 hr | 12,672             | 14,004           |
| LOS in ICU > 3 days             | Admission + 48 hr | 12,712             | 14,064           |
| LOS in Hospital > 3 days        | Admission + 24 hr | 99,540             | 99,540           |
| LOS in Hospital > 3 days        | Admission + 48 hr | 99,786             | 99,786           |
