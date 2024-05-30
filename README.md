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

### Scripts and Examples

See `tests/test_tabularize_integration.py` for an example of the end-to-end pipeline being run on synthetic data. This
script is a functional test that is also run with `pytest` to verify correctness of the algorithm.

#### Core Scripts:

1. `scripts/tabularize/identify_columns.py` loads all training shard to identify which feature columns
   to generate tabular data for.

```bash
POLARS_MAX_THREADS=32 python scripts/identify_columns.py MEDS_cohort_dir=/storage/shared/meds_tabular_ml/ebcl_dataset/processed/final_cohort tabularized_data_dir=/storage/shared/meds_tabular_ml/ebcl_dataset/processed/tabularize min_code_inclusion_frequency=1 "window_sizes=[1d, 7d, full]" do_overwrite=True
```

2. `scripts/tabularize/tabularize_static.py` Iterates through shards and generates tabular vectors for
   each patient. There is a single row per patient for each shard.

```bash
POLARS_MAX_THREADS=32 python scripts/tabularize_static.py MEDS_cohort_dir=/storage/shared/meds_tabular_ml/ebcl_dataset/processed/final_cohort tabularized_data_dir=/storage/shared/meds_tabular_ml/ebcl_dataset/processed/tabularize min_code_inclusion_frequency=1 "window_sizes=[1d, 7d, full]" do_overwrite=True
```

4. `scripts/tabularize/summarize_over_windows.py` For each shard, iterates through window sizes and aggregations to
   and horizontally concatenates the outputs to generate the final tabular representations at every event time for every patient.

```bash
POLARS_MAX_THREADS=1 python scripts/summarize_over_windows.py MEDS_cohort_dir=/storage/shared/meds_tabular_ml/ebcl_dataset/processed/final_cohort tabularized_data_dir=/storage/shared/meds_tabular_ml/ebcl_dataset/processed/tabularize min_code_inclusion_frequency=1 "window_sizes=[1d, 7d, full]" do_overwrite=True
```

## Feature Construction, Storage, and Loading

Tabularization of a (raw) MEDS dataset is done by running the `scripts/data/tabularize.py` script. This script
must inherently do a base level of preprocessing over the MEDS data, then will construct a sharded tabular
representation that respects the overall sharding of the raw data. This script uses [Hydra](https://hydra.cc/)
to manage configuration, and the configuration file is located at `configs/tabularize.yaml`.

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

# YAML Configuration File

- `MEDS_cohort_dir`: directory of MEDS format dataset that is ingested.
- `tabularized_data_dir`: output directory of tabularized data.
- `min_code_inclusion_frequency`: The base feature inclusion frequency that should be used to dictate
  what features can be included in the flat representation. It can either be a float, in which
  case it applies across all measurements, or `None`, in which case no filtering is applied, or
  a dictionary from measurement type to a float dictating a per-measurement-type inclusion
  cutoff.
- `window_sizes`: Beyond writing out a raw, per-event flattened representation, the dataset also has
  the capability to summarize these flattened representations over the historical windows
  specified in this argument. These are strings specifying time deltas, using this syntax:
  `link`\_. Each window size will be summarized to a separate directory, and will share the same
  subject file split as is used in the raw representation files.
- `codes`: A list of codes to include in the flat representation. If `None`, all codes will be included
  in the flat representation.
- `aggs`: A list of aggregations to apply to the raw representation. Must have length greater than 0.
- `n_patients_per_sub_shard`: The number of subjects that should be included in each output file.
  Lowering this number increases the number of files written, making the process of creating and
  leveraging these files slower but more memory efficient.
- `do_overwrite`: If `True`, this function will overwrite the data already stored in the target save
  directory.
- `seed`: The seed to use for random number generation.
