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

For an end-to-end example over MIMIC-IV, see the [MIMIC-IV companion repository](https://github.com/mmcdermott/MEDS_TAB_MIMIC_IV).
For an end-to-end example over Philips eICU, see the [eICU companion repository](https://github.com/mmcdermott/MEDS_TAB_EICU).

See [`/tests/test_integration.py`](https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/main/tests/test_integration.py) for a local example of the end-to-end pipeline being run on synthetic data. This script is a functional test that is also run with `pytest` to verify the correctness of the algorithm.

## Core CLI Scripts Overview

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

   **Example: Align tabularized data** for a specific task `$TASK` and labels that has pulled from [ACES](https://github.com/justin13601/ACES)

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
