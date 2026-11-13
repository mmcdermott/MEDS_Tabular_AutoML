# Scalable tabularization and tabular feature usage utilities over generic MEDS datasets

<p align="center">
  <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/-Python_3.11+-blue?logo=python&logoColor=white"></a>
  <a href="https://pypi.org/project/meds-tab/"><img alt="PyPI" src="https://img.shields.io/pypi/v/meds-tab?color=orange"></a>
  <a href="https://hydra.cc/"><img alt="Hydra" src="https://img.shields.io/badge/Config-Hydra_1.3-89b8cd"></a>
  <a href="https://codecov.io/gh/mmcdermott/MEDS_Tabular_AutoML"><img alt="Codecov" src="https://codecov.io/gh/mmcdermott/MEDS_Tabular_AutoML/graph/badge.svg?token=6GD05EDQ39"></a>
  <a href="https://github.com/mmcdermott/MEDS_Tabular_AutoML/actions/workflows/tests.yaml"><img alt="Tests" src="https://github.com/mmcdermott/MEDS_Tabular_AutoML/actions/workflows/tests.yaml/badge.svg"></a>
  <a href="https://github.com/mmcdermott/MEDS_Tabular_AutoML/actions/workflows/code-quality-main.yaml"><img alt="Code Quality" src="https://github.com/mmcdermott/MEDS_Tabular_AutoML/actions/workflows/code-quality-main.yaml/badge.svg"></a>
  <a href='https://meds-tab.readthedocs.io/en/latest/?badge=latest'><img src='https://readthedocs.org/projects/meds-tab/badge/?version=latest' alt='Documentation Status' /></a>
  <a href="https://github.com/mmcdermott/MEDS_Tabular_AutoML/graphs/contributors"><img alt="Contributors" src="https://img.shields.io/github/contributors/mmcdermott/MEDS_Tabular_AutoML.svg"></a>
  <a href="https://github.com/mmcdermott/MEDS_Tabular_AutoML/pulls"><img alt="Pull Requests" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg"></a>
  <a href="https://github.com/mmcdermott/MEDS_Tabular_AutoML#license"><img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray"></a>
</p>

# Welcome!

MEDS-Tab turns longitudinal electronic health records in the [Medical Event Data Standard (MEDS)](https://github.com/Medical-Event-Data-Standard/meds) into wide, sparse tabular features (counts, sums, minimums, maximums and more over several time windows), and trains strong baseline models on them for any binary prediction task.

## Why MEDS-Tab?

- **Fast:** the native C++ engine tabularizes a whole MIMIC-IV task in about a minute on a laptop, and `medsboost`, MEDS-Tab's own gradient-boosted trees for these matrices, trains in a few minutes.
- **Strong baselines:** on 40 MIMIC-IV and NWICU tasks of a recent benchmark of 12 models ([arXiv 2609.18134](https://arxiv.org/abs/2609.18134)), one untuned `medsboost` model per task matches the benchmark's strongest model, MEDS-Tab-large, on average (see [Results](#results)).
- **Any MEDS dataset, any task:** tasks are defined by label files (for example from [ACES](https://github.com/justin13601/ACES) or [MEDS-DEV](https://github.com/Medical-Event-Data-Standard/MEDS-DEV)); features are computed at each label's prediction time.
- **Runs anywhere:** compiled wheels for Linux, Windows and Apple-silicon macOS; a pure-Python fallback elsewhere.

### I. Transform to MEDS

MEDS-Tab reads data in the MEDS format. To convert your raw EHR data, use:

- [MEDS Transforms](https://github.com/mmcdermott/MEDS_transforms), functions and scripts to extract and pre-process MEDS data;
- [MEDS ETL](https://github.com/Medical-Event-Data-Standard/meds_etl), ETLs from common formats (MIMIC-IV, OMOP v5, MEDS FLAT);
- dataset-specific ETLs such as [MIMIC-IV MEDS](https://github.com/Medical-Event-Data-Standard/MIMIC_IV_MEDS).

### II. Run MEDS-Tab

Describe the codes, tabularize a task, and train a model with four commands (see [Quick Start](#quick-start)), then read the [Usage Guide](https://meds-tab.readthedocs.io/en/latest/usage_guide/) and the [MIMIC-IV tutorial](https://github.com/mmcdermott/MEDS_Tabular_AutoML/tree/main/MIMICIV_TUTORIAL) for every option.

______________________________________________________________________

# Usage

This repository has two parts:

1. Construction and efficient loading of tabular (flat, non-longitudinal) summary features describing each subject's record over time windows that end at a label's prediction time (e.g. the last day, 30 days, year, or the full history).
2. An AutoML pipeline over these features for binary classification tasks: by default a hyperparameter sweep of `medsboost` (called `sparsegb` in 0.4.0), with XGBoost, LightGBM and scikit-learn models also available. The search itself is simple; what MEDS-Tab makes fast is building, storing and loading very wide feature sets, so far more features can be searched over than in prior systems.

## Quick Start

### Install

```console
pip install meds-tab
```

On Linux (x86_64 and aarch64), Windows (x86_64) and Apple-silicon Macs with macOS 15 or later, for Python
3.11 to 3.13, this installs a compiled wheel that includes `medsboost`, the default and recommended model
launcher, and the native tabularization engine. Check with
`python -c "from MEDS_tabular_automl import medsboost; print(medsboost.is_available())"`.

On other platforms or Python versions, pip installs a pure-Python wheel without them: tabularization then
uses its Python implementation, and you pick another model with `model_launcher=lightgbm` or
`model_launcher=xgboost`. To build them there, install from source with a C++17 compiler and OpenMP (GCC on
Linux; Apple clang after `brew install libomp` on macOS; Visual Studio's C++ build tools on Windows):

```console
pip install --no-binary meds-tab meds-tab
# or, from a clone of this repository:
pip install .
```

### Run

With MEDS data in `$MEDS_DIR/data` and a task's labels in `$LABELS_DIR` (`<split>/<shard>.parquet`, one file
per data shard):

```console
meds-tab-describe "input_dir=$MEDS_DIR/data" "output_dir=$OUTPUT_DIR"
meds-tab-tabularize-static "input_dir=$MEDS_DIR/data" "output_dir=$OUTPUT_DIR" "input_label_dir=$LABELS_DIR"
meds-tab-tabularize-time-series "input_dir=$MEDS_DIR/data" "output_dir=$OUTPUT_DIR" "input_label_dir=$LABELS_DIR"
meds-tab-model --multirun "input_dir=$MEDS_DIR/data" "output_dir=$OUTPUT_DIR" "input_label_dir=$LABELS_DIR" \
    "output_model_dir=$MODEL_DIR" "task_name=$TASK" "hydra.sweeper.n_trials=100"
```

This uses the default windows (`1d`, `7d`, `30d`, `365d`, full history) and aggregations; set
`tabularization.window_sizes` and `tabularization.aggs` to change them. Drop `--multirun` to train a single
model with default hyperparameters. Models, predictions and tuning and held-out AUROCs are written under
`$MODEL_DIR`.

## Scripts and Examples

For an end-to-end example over MIMIC-IV, see the [MIMIC-IV tutorial](https://github.com/mmcdermott/MEDS_Tabular_AutoML/tree/main/MIMICIV_TUTORIAL).

See [`/tests/test_integration.py`](https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/main/tests/test_integration.py) for the whole pipeline run on synthetic data. It is a functional test, also run with `pytest`.

## Results

Held-out AUROC of one `medsboost` model per task, with default hyperparameters (no sweep), against the 12
models of the benchmark in [arXiv 2609.18134](https://arxiv.org/abs/2609.18134) (Tables 9–12). Features:
7 windows (2 hours to the full history) × all aggregations, codes seen at least 10 times. Run on a laptop with
6 cores and 61 GB of memory.

| Task suite              | Tasks | `medsboost`, mean AUROC | MEDS-Tab-large, mean AUROC | Tasks where `medsboost` ranks first of 13 |
| ----------------------- | ----: | ----------------------: | -------------------------: | ----------------------------------------: |
| MIMIC-IV clinical (MC)  |    10 |                   0.862 |                      0.860 |                                         8 |
| MIMIC-IV generated (MG) |    10 |                   0.953 |                      0.951 |                                         6 |
| NWICU clinical (NC)     |    10 |                   0.816 |                      0.818 |                                         3 |
| NWICU generated (NG)    |    10 |                   0.845 |                      0.845 |                                         4 |

Each MIMIC-IV task took about 1 minute to tabularize and 3 to 6 minutes to train, with a peak of 11 to 51 GB
of memory during training; each NWICU task took about 30 seconds in total. These are single runs without
confidence intervals, and the benchmark's own numbers are point estimates: read per-task differences below
about 0.01 as noise.

# The MEDS-Tab Architecture

MEDS-Tab addresses two challenges in healthcare machine learning: efficiently tabularizing large-scale electronic health record (EHR) data, and training competitive baseline models on it. It scales to hundreds of millions of events and hundreds of thousands of feature columns through:

- sparse data structures throughout, from tabularization to model training;
- sharding, so each piece of work fits in memory and runs independently;
- a native C++ engine for tabularization and a gradient-boosting library written for these sparse matrices;
- configurable processing parameters for different hardware.

## Overview

The pipeline has five stages, the first optional:

1. **Resharding** (optional) restructures the input data into manageable shards.
2. **Data description** (`meds-tab-describe`) computes per-code statistics on the training subjects, used to select features.
3. **Static** and
4. **time-series tabularization** (`meds-tab-tabularize-static`, `meds-tab-tabularize-time-series`) compute task-specific features: one row per label of a prediction task, with every window ending at the label's prediction time.
5. **Model training** (`meds-tab-model`) trains one model or runs a hyperparameter sweep, and evaluates the best model on the held-out split.

Versions up to 0.2.0 could also tabularize every event time once and then cache each task's rows (`meds-tab-cache-task`); that workflow has been removed (see the [usage guide](https://meds-tab.readthedocs.io/en/latest/usage_guide/)).

## Memory Management Via Sparse Data Structures

Most codes are absent for any given subject at any given time, so features are stored as sparse matrices (`scipy.sparse`, and `.npz` blocks on disk), one block per shard, window and aggregation. Shards are processed independently, which keeps memory bounded and lets work run in parallel. In our MIMIC-IV runs, tabularization needed about 2 GB of memory for every task; model training loads the task's matrices, so its memory grows with the number of labels and features.

## Native Backend

Static and time-series tabularization run on a C++ engine by default: the setting `backend=auto` uses it
whenever it is built and supports the configuration, and otherwise uses the Python implementation, logging
why. The engine is included in the compiled wheels (Linux, Windows and Apple-silicon macOS) and built by a
source install. It writes the same files as the Python implementation, byte for byte, so `TabularDataset` and
the model launchers are unchanged. It is much faster: on a 10% MIMIC-IV cohort it tabularizes in about
12 seconds, against 26.5 minutes for the Python implementation with 16 workers. Force one implementation with
`backend=native` or `backend=python`, or with the `MEDS_TAB_BACKEND` environment variable. See the
[native backend guide](docs/native_backend.md).

## Feature Engineering Via Rolling Windows and Aggregation Functions

Static features capture the presence of each code and, for codes with static numeric values, the mean of the subject's values (`static/mean`). Time-series features aggregate codes and numeric values (counts, sums, sums of squares, minimums and maximums) over windows such as 2 hours, 1 day, 30 days, 365 days or the full history, ending at each label's prediction time. Codes can be filtered by frequency in the training data (`tabularization.min_code_inclusion_count` and related options), and the model stage can treat these thresholds as hyperparameters.

## Model Support and Normalization/Imputation Options

The default and recommended model is `medsboost`, MEDS-Tab's own gradient-boosted trees for wide, sparse tabularized data. It runs the same algorithm as LightGBM (Newton boosting, histograms, leaf-wise growth, bagging, early stopping), but never stores or scans absent entries, drops features that can no longer split, and samples one window per code and aggregation in each tree. On the 10 MIMIC-IV MG tasks it trains 12.5 times faster than LightGBM at the same held-out AUROC. During a sweep, each split's matrix is cached on its first load, so later trials skip re-reading the tabularized files.

XGBoost and LightGBM are also supported, as are KNN, logistic regression, random forest and SGD classifiers from scikit-learn. An experimental AutoGluon integration provides automated model selection.

Normalization options (standard scaler, max-abs scaler) preserve sparsity, while imputation methods (mean, median, mode) are available when dense representations help other models.

## Additional Design Considerations

**Extensibility and maintainability**: The pipeline's modular design allows for the addition of new feature types, aggregation methods, and model support. Contributions are welcome!

**Highly Configurable**: Every stage is configured with Hydra; see the usage guide for the parameters.

## Roadmap

MEDS-Tab has several key limitations which we plan to address in future changes. These include, and are tracked by, the following GitHub issues.

### Improvements to the core tabularization

1. Further memory and runtime improvements are possible: [#16](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/16)
2. We should support additional window sizes and types: [#31](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/31)
3. We should support additional aggregation functions: [#32](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/32)

### Improvements to the modeling pipeline

1. We should likely decorrelate the default aggregations and/or window sizes we use prior to passing them into the models as features: [#27](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/27)
2. We should support a more extensive set of pipeline operations and model architectures: [#37](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/37)
3. `medsboost` handles binary tasks only; multiclass and regression are not supported yet.

### Technical debt / code improvements

1. The computation and use of the code metadata dataframe, containing frequencies of codes, should be offloaded to core MEDS functionality, with the remaining code in this repository cleaned up.
    - [#28](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/28)
2. We should add more doctests and push test coverage up to 100%
    - [#29](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/29)
    - [#30](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/30)

## What do you mean "tabular pipelines"? Isn't _all_ structured EHR data already tabular?

This is a common misconception. _Tabular_ data refers to data that can be organized in a consistent, logical
set of rows/columns such that the entirety of a "sample" or "instance" for modeling or analysis is contained
in a single row, and the set of columns possibly observed (there can be missingness) is consistent across all
rows. Structured EHR data does not satisfy this definition, as we will have different numbers of observations
of medical codes and values at different timestamps for different patients, so it cannot simultaneously
satisfy the (1) "single row single instance", (2) "consistent set of columns", and (3) "logical" requirements.
Thus, in this pipeline, when we say we will produce a "tabular" view of MEDS data, we mean a dataset that can
realize these constraints, which will explicitly involve summarizing the patient data over various historical
or future windows in time to produce a single row per patient with a consistent, logical set of columns
(though there may still be missingness).
