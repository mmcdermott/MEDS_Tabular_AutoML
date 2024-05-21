# Scalable tabularization and tabular feature usage utilities over generic MEDS datasets
This repository provides utilities and scripts to run limited automatic tabular ML pipelines for generic MEDS
datasets.

Why not other systems?
  - [TemporAI](https://github.com/vanderschaarlab/temporai) is the most natural competitor, and already
    supports AutoML capabilities. However, TemporAI (as of now) does not support generic MEDS datasets, and it
    is not clear if their AutoML systems will scale to the size of datasets we need to support. But, further
    investigation is needed, and it may be the case that the best solution here is simply to write a custom
    data source for MEDS data within TemporAI and leverage their tools.

# Installation
Clone this repository and install the requirements by running `pip install .` in the root directory.

# Usage
This repository consists of two key pieces:
  1. Construction of and efficient loading of tabular (flat, non-longitudinal) summary features describing
     patient records in MEDS over arbitrary time-windows (e.g. 1 year, 6 months, etc.) either backwards or
     forwards in time from a given index date. Naturally, only "look-back" windows should be used for
     future-event prediction tasks; however, the capability to summarize "look-ahead" windows is also useful
     for characterizing and describing the differences between patient populations statistically.
  2. Running basic AutoML pipelines over these tabular features to predict arbitrary binary classification
     downstream tasks defined over these datasets. The "AutoML" part of this is not particularly advanced --
     what is more advanced is the efficient construction, storage, and loading of tabular features for the
     candidate AutoML models, enabling a far more extensive search over different featurization strategies.

## Feature Construction, Storage, and Loading

## AutoML Pipelines

# TODOs
  1. Leverage the "event bound aggregation" capabilities of [ESGPT Task
     Select](https://github.com/justin13601/ESGPTTaskQuerying/) to construct tabular summary features for
     event-bound historical windows (e.g., until the prior admission, until the last diagnosis of some type,
     etc.).
  2. Support more feature aggregation functions.
  3. Probably rename this repository, as the focus is really more on the tabularization and feature usage
     utilities than on the AutoML pipelines themselves.
  4. Import, rather than reimplement, the mapper utilities from the MEDS preprocessing repository.
  5. Investigate the feasibility of using TemporAI for this task.
  6. Consider splitting the feature construction and AutoML pipeline parts of this repository into separate
     repositories.
