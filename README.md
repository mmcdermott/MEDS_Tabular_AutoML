# Scalable tabularization and tabular feature usage utilities over generic MEDS datasets

This repository provides utilities and scripts to run limited automatic tabular ML pipelines for generic MEDS
datasets.

#### Q1: What do you mean "tabular pipelines"? Isn't _all_ structured EHR data already tabular?

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

#### Q2: Why not other systems?

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

Tabularization of a (raw) MEDS dataset is done by running the `scripts/data/tabularize.py` script. This script
must inherently do a base level of preprocessing over the MEDS data, then will construct a sharded tabular
representation that respects the overall sharding of the raw data. This script uses [Hydra](https://hydra.cc/)
to manage configuration, and the configuration file is located at `configs/tabularize.yaml`.

Tabularization will take as input a MEDS dataset in a directory we'll denote `$MEDS_cohort_dir` and will write out a collection of tabularization files to disk in subdirectories of this cohort directory. In particular for a given shard prefix in the raw MEDS cohort (e.g., `train/0`, `held_out/1`, etc.)

1. In `$MEDS_cohort_dir/tabularized/static/$SHARD_PREFIX.parquet` will be tabularized, wide-format representations of code / value occurrences with null timestamps. In the case that sub-sharding is needed, sub-shards will instead be written as sub-directories of this base directory: `$MEDS_cohort_dir/tabularized/static/$SHARD_PREFIX/$SUB_SHARD.parquet`. This sub-sharding pattern will hold for all files and not be subsequently measured.
2. In `$MEDS_cohort_dir/tabularized/at_observation/$SHARD_PREFIX.parquet` will be tabularized, wide-format representations of code / value observations for all observations of patient data with a non-null timestamp.
3. In `$MEDS_cohort_dir/tabularized/over_window/$WINDOW_SIZE/$SHARD_PREFIX.parquet` will be tabularized, wide-format summarization of the code / value occurrences over a window of size `$WINDOW_SIZE` as of the index date at the row's timestamp.

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
