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

MEDS-Tab is a comprehensive framework designed to streamline the handling, modeling, and analysis of complex medical time-series data. By leveraging automated processes, MEDS-Tab significantly reduces the computation required to generate high-quality baseline models for diverse supervised learning tasks.

- Cost Efficiency: MEDS-Tab is dramatically more cost-effective compared to existing solutions
- Strong Performance: MEDS-Tab provides robustness and high performance across various datasets compared with other frameworks.

### I. Transform to MEDS

MEDS-Tab leverages the recently developed, minimal, easy-to-use Medical Event Data Standard (MEDS) schema to standardize structured EHR data to a consistent schema from which baselines can be reliably produced across arbitrary tasks and settings. In order to use MEDS-Tab, you will first need to transform your raw EHR data to a MEDS format, which can be done using the following libraries:

- [MEDS Transforms](https://github.com/mmcdermott/MEDS_transforms) for a set of functions and scripts for extraction to and transformation/pre-processing of MEDS-formatted data.
- [MEDS ETL](https://github.com/Medical-Event-Data-Standard/meds_etl) for a collection of ETLs from common data formats to MEDS. The package library currently supports MIMIC-IV, OMOP v5, and MEDS FLAT (a flat version of MEDS).

### II. Run MEDS-Tab

- Run the MEDS-Tab Command-Line Interface tool (`MEDS-Tab-cli`) to extract cohorts based on your task - check out the [Usage Guide](https://meds-tab--36.org.readthedocs.build/en/36/overview.html#core-cli-scripts-overview) and the [MIMIC-IV tutorial](https://github.com/mmcdermott/MEDS_Tabular_AutoML/tree/main/MIMICIV_TUTORIAL)!

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
