# Scalable tabularization and tabular feature usage utilities over generic MEDS datasets

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3100/"><img alt="Python" src="https://img.shields.io/badge/-Python_3.12+-blue?logo=python&logoColor=white"></a>
  <a href="https://pypi.org/project/meds-tab/"><img alt="PyPI" src="https://img.shields.io/badge/PyPI-v0.2.5-orange?logoColor=orange"></a>
  <a href="https://hydra.cc/"><img alt="Hydra" src="https://img.shields.io/badge/Config-Hydra_1.3-89b8cd"></a>
  <a href="https://codecov.io/gh/mmcdermott/MEDS_Tabular_AutoML"><img alt="Codecov" src="https://codecov.io/gh/mmcdermott/MEDS_Tabular_AutoML/graph/badge.svg?token=6GD05EDQ39"></a>
  <a href="https://github.com/mmcdermott/MEDS_Tabular_AutoML/actions/workflows/tests.yaml"><img alt="Tests" src="https://github.com/mmcdermott/MEDS_Tabular_AutoML/actions/workflows/tests.yaml/badge.svg"></a>
  <a href="https://github.com/mmcdermott/MEDS_Tabular_AutoML/actions/workflows/code-quality-main.yaml"><img alt="Code Quality" src="https://github.com/mmcdermott/MEDS_Tabular_AutoML/actions/workflows/code-quality-main.yaml/badge.svg"></a>
  <!--<a href="https://eventstreamaces.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation" src="https://readthedocs.org/projects/eventstreamaces/badge/?version=latest"/></a>-->
  <a href="https://github.com/mmcdermott/MEDS_Tabular_AutoML/graphs/contributors"><img alt="Contributors" src="https://img.shields.io/github/contributors/mmcdermott/MEDS_Tabular_AutoML.svg"></a>
  <a href="https://github.com/mmcdermott/MEDS_Tabular_AutoML/pulls"><img alt="Pull Requests" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
  <a href="https://github.com/mmcdermott/MEDS_Tabular_AutoML#license"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray"></a>
</p>

This repository provides utilities and scripts to run limited automatic tabular ML pipelines for generic MEDS
datasets.

# Installation

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

1. Construction of and efficient loading of tabular (flat, non-longitudinal) summary features describing
   patient records in MEDS over arbitrary time-windows (e.g. 1 year, 6 months, etc.) either backwards or
   forwards in time from a given index date. Naturally, only "look-back" windows should be used for
   future-event prediction tasks, and are thus currently implemented.
2. Running a basic XGBoost AutoML pipeline over these tabular features to predict arbitrary binary classification or regression
   downstream tasks defined over these datasets. The "AutoML" part of this is not particularly advanced --
   what is more advanced is the efficient construction, storage, and loading of tabular features for the
   candidate AutoML models, enabling a far more extensive search over different featurization strategies.

### Scripts and Examples

See `tests/test_integration.py` for an example of the end-to-end pipeline being run on synthetic data. This
script is a functional test that is also run with `pytest` to verify the correctness of the algorithm.

For an end to end example over MIMIC-IV, see the [companion repository](https://github.com/mmcdermott/MEDS_TAB_MIMIC_IV)

### Core CLI Scripts Overview

1. **`meds-tab-describe`**: This command processes MEDS data shards to compute the frequencies of different code-types

   - time-series codes (codes with timestamps)
   - time-series numerical values (codes with timestamps and numerical values)
   - static codes (codes without timestamps)
   - static numerical codes (codes without timestamps but with numerical values).

2. **`meds-tab-tabularize-static`**: Filters and processes the dataset based on the frequency of codes, generating a tabular vector for each patient at each timestamp in the shards. Each row corresponds to a unique `patient_id` and `timestamp` combination, thus rows are duplicated across multiple timestamps for the same patient.

3. **`meds-tab-tabularize-time-series`**: Iterates through combinations of a shard, `window_size`, and `aggregation` to generate feature vectors that aggregate patient data for each unique `patient_id` x `timestamp`. This stage (and the previous stage) use sparse matrix formats to efficiently handle the computational and storage demands of rolling window calculations on large datasets. We support parallelization through Hydra's `--multirun` flag and the `joblib` launcher.

4. **`meds-tab-cache-task`**: Aligns task-specific labels with the nearest prior event in the tabularized data. It requires a labeled dataset directory with three columns (`patient_id`, `timestamp`, `label`) structured similarly to the `MEDS_cohort_dir`.

5. **`meds-tab-xgboost`**: Trains an XGBoost model using user-specified parameters.

6. **`meds-tab-xgboost-sweep`**: Conducts an Optuna hyperparameter sweep to optimize over `window_sizes`, `aggregations`, and `min_code_inclusion_frequency`, aiming to enhance model performance and adaptability.

# How does MEDS-Tab Work?

#### What do you mean "tabular pipelines"? Isn't _all_ structured EHR data already tabular?

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

## Implementation Improvements

# Computational Performance vs. Existing Pipelines

# XGBoost Performance

## XGBoost Model Performance on MIMIC-IV

## XGBoost Model Performance on Philips eICU
