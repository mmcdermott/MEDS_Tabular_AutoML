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

# Welcome!

MEDS-Tab is a library designed for automated tabularization, data preparation with aggregations and time windowing. Check out below for an overview of MEDS-Tab and how it could be useful in your workflows!

## Why MEDS-Tab?

MEDS-Tab is a tool for tabularization and associated modeling of complex medical time-series data. By leveraging sparse representations during tabularization and parallelism over shards, MEDS-Tab significantly reduces the computation required to generate high-quality baseline models for diverse supervised learning tasks.

- Cost Efficiency: MEDS-Tab is dramatically more cost-effective compared to existing solutions
- Strong Performance: MEDS-Tab provides robustness and high performance across various datasets compared with other frameworks.

### I. Transform to MEDS

MEDS-Tab leverages the recently developed, minimal, easy-to-use Medical Event Data Standard (MEDS) schema to standardize structured EHR data to a consistent schema from which baselines can be reliably produced across arbitrary tasks and settings. In order to use MEDS-Tab, you will first need to transform your raw EHR data to a MEDS format, which can be done using the following libraries:

- [MEDS Transforms](https://github.com/mmcdermott/MEDS_transforms) for a set of functions and scripts for extraction to and transformation/pre-processing of MEDS-formatted data.
- [MEDS ETL](https://github.com/Medical-Event-Data-Standard/meds_etl) for a collection of ETLs from common data formats to MEDS. The package library currently supports MIMIC-IV, OMOP v5, and MEDS FLAT (a flat version of MEDS).

### II. Run MEDS-Tab

- Run the MEDS-Tab Command-Line Interface tool (`MEDS-Tab-cli`) to extract cohorts based on your task - check out the [Usage Guide](https://meds-tab.readthedocs.io/en/latest/usage_guide/) and the [MIMIC-IV tutorial](https://github.com/mmcdermott/MEDS_Tabular_AutoML/tree/main/MIMICIV_TUTORIAL)!

- Painless Reproducibility: Use [MEDS-Tab](https://github.com/mmcdermott/MEDS_Tabular_AutoML/tree/main/MIMICIV_TUTORIAL) to obtain comparable, reproducible, and well-tuned XGBoost results tailored to your dataset-specific feature space!

By following these steps, you can seamlessly transform your dataset, define necessary criteria, and leverage powerful machine learning tools within the MEDS-Tab ecosystem. This approach not only simplifies the process but also ensures high-quality, reproducible results for your machine learning tasks for health projects. It can reliably take no more than a week of full-time human effort to perform Steps I-V on new datasets in reasonable raw formulations!

______________________________________________________________________

# Usage

This repository consists of two key pieces:

1. Construction and efficient loading of tabular (flat, non-longitudinal) summary features describing patient records in MEDS over arbitrary time windows (e.g. 1 year, 6 months, etc.), which go backward in time from a given index date.
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

For an end-to-end example over MIMIC-IV, see the [MIMIC-IV tutorial](https://github.com/mmcdermott/MEDS_Tabular_AutoML/tree/main/MIMICIV_TUTORIAL).

See [`/tests/test_integration.py`](https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/main/tests/test_integration.py) for a local example of the end-to-end pipeline being run on synthetic data. This script is a functional test that is also run with `pytest` to verify the correctness of the algorithm.

# The MEDS-Tab Architecture

MEDS-Tab addresses two key challenges in healthcare machine learning: efficiently tabularizing large-scale electronic health record (EHR) data and training competitive baseline models on this tabularized data. This document outlines the architecture and implementation details of MEDS-Tab's pipeline.

MEDS-Tab is designed to scale to hundreds of millions of events and tens of thousands of unique medical codes. Performance optimization is achieved through:

- Efficient parallel processing when appropriate
- Strategic use of sparse data structures
- Memory-aware data loading and processing
- Configurable processing parameters for different hardware capabilities

## Overview

The MEDS-Tab pipeline consists of six main stages, with the first being optional. The pipeline begins with an optional (1) data resharding stage that optimizes processing by restructuring input data into manageable chunks. This is followed by (2) data description, which computes some summary statistics over the features in the dataset. The core processing happens in the (3) static and (4) time-series tabularization stages, which transform the data into a format suitable for tabular machine learning. (5) Task-specific data caching then aligns this data with prediction tasks, and finally, the (6) model training stage provides efficient training capabilities with support for multiple model types and hyperparameter optimization.

## Memory Management Via Sparse Data Structures

Memory management is a central consideration in MEDS-Tab's design. The system employs several key strategies to handle large-scale medical datasets efficiently:

Sparse matrix operations form the foundation of our memory management approach. We utilize scipy.sparse for memory-efficient storage of sparse non-zero elements, which is particularly effective for medical data where most potential features are not present for any given patient at any given time.

Data sharding complements our sparse matrix approach by breaking data into manageable chunks. This enables both memory-efficient processing and parallelization. Shards are processed independently, allowing us to handle datasets that would be impossible to process as a single unit.

The system implements efficient aggregation using Polars for fast rolling window computations. This optimizes same-day event aggregation and maintains memory efficiency during temporal calculations.

## Improved Computational Speed Via Parallel Processing

Our processing strategy differentiates between sequential and parallel operations based on computational needs and data dependencies. The data description and static tabularization stages operate sequentially, as they have manageable computational requirements. In contrast, time-series tabularization, task-specific caching, and model training leverage parallel processing over independent workers (which may be spawned on different cores on a local machine or over a slurm cluster) to handle their more intensive computational demands.

Data flow through the pipeline is optimized through caching and sharding. Each stage's output is structured to minimize memory requirements while maintaining accessibility for subsequent stages. The system preserves sparsity wherever possible and uses efficient shard management to increase processing speed and reduce total memory consumption.

## Feature Engineering Via Rolling Windows and Aggregation Functions

MEDS-Tab implements a comprehensive feature engineering approach that handles both static and temporal data. For static features, we capture both presence and first-recorded values (as there should be only one occurrence of a static code). Time-series features are processed through various aggregation methods including counts, sums, minimums, and maximums. These aggregations can be computed over multiple time windows (1 day, 30 days, 365 days, or the full patient history), providing temporal context at different scales.

Our feature engineering framework maintains flexibility while enforcing consistency. All aggregations preserve sparsity where possible, and the system includes configurable thresholds for feature inclusion based on frequency and relevance to the target task.

## Model Support and Normalization/Imputation Options

The architecture includes robust support for multiple model types, with XGBoost as the primary implementation. Additional supported models include KNN Classifier, Logistic Regression, Random Forest Classifier, and SGD Classifier. An experimental AutoGluon integration provides automated model selection and tuning capabilities.

Data processing options are designed to maintain efficiency while providing necessary transformations. Normalization options (standard scaler, max abs scaler) preserve sparsity, while imputation methods (mean, median, mode) are available when dense representations are required or beneficial.

## Additional Design Considerations

**Extensibility and maintainability**: The pipeline's modular design allows for the addition of new feature types, aggregation methods, and model support. Contributions are welcome!

**Highly Configurable**: This pipeline is highly configurable via parameters that allow users to adjust processing based on their specific needs and hardware constraints. See the usage guide for more details.

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
2. We should add more doctests and push test coverage up to 100%
    - [#29](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/29)
    - [#30](https://github.com/mmcdermott/MEDS_Tabular_AutoML/issues/30)

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
