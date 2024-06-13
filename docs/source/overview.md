# Overview

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
For an end to end example over Philips eICU, see the [eICU companion repository](https://github.com/mmcdermott/MEDS_TAB_EICU).

### Core CLI Scripts Overview

1. **`meds-tab-describe`**: This command processes MEDS data shards to compute the frequencies of different code-types

   - time-series codes (codes with timestamps)
   - time-series numerical values (codes with timestamps and numerical values)
   - static codes (codes without timestamps)
   - static numerical codes (codes without timestamps but with numerical values).

   **Caching feature names and frequencies** in a dataset stored in `"path_to_data"`

   ```
   meds-tab-describe MEDS_cohort_dir="path_to_data"
   ```

2. **`meds-tab-tabularize-static`**: Filters and processes the dataset based on the frequency of codes, generating a tabular vector for each patient at each timestamp in the shards. Each row corresponds to a unique `patient_id` and `timestamp` combination, thus rows are duplicated across multiple timestamps for the same patient.

   **Tabularizing static data** with the minimum code frequency of 10 and window sizes of `[1d, 30d, 365d, full]` and value aggregation methods of `[static/present, code/count, value/count, value/sum, value/sum_sqd, value/min, value/max]`

   ```
   meds-tab-tabularize-static MEDS_cohort_dir="path_to_data" \
                               tabularization.min_code_inclusion_frequency=10 \
                               tabularization.window_sizes=[1d,30d,365d,full] \
                               do_overwrite=False \
                               tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]"
   ```

3. **`meds-tab-tabularize-time-series`**: Iterates through combinations of a shard, `window_size`, and `aggregation` to generate feature vectors that aggregate patient data for each unique `patient_id` x `timestamp`. This stage (and the previous stage) use sparse matrix formats to efficiently handle the computational and storage demands of rolling window calculations on large datasets. We support parallelization through Hydra's `--multirun` flag and the `joblib` launcher.

   **Aggregates time-series data** on features across different `window_sizes`

   ```
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

   **Aligh tabularized data** for a specific task `$TASK` and labels that has pulled from [ACES](https://github.com/justin13601/ACES)

   ```
   meds-tab-cache-task MEDS_cohort_dir="path_to_data" \
                       task_name=$TASK \
                       tabularization.min_code_inclusion_frequency=10 \
                       do_overwrite=False \
                       tabularization.window_sizes=[1d,30d,365d,full] \
                       tabularization.aggs=[static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max]
   ```

5. **`meds-tab-xgboost`**: Trains an XGBoost model using user-specified parameters. Permutations of `window_sizes` and `aggs` can be generated using `generate-permutations` command (See the section below for descriptions).

   ```
   meds-tab-xgboost --multirun \
                     MEDS_cohort_dir="path_to_data" \
                     task_name=$TASK \
                     output_dir="output_directory" \
                     tabularization.min_code_inclusion_frequency=10 \
                     tabularization.window_sizes=$(generate-permutations [1d,30d,365d,full]) \
                     do_overwrite=False \
                     tabularization.aggs=$(generate-permutations [static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max])
   ```

6. **`meds-tab-xgboost-sweep`**: Conducts an Optuna hyperparameter sweep to optimize over `window_sizes`, `aggregations`, and `min_code_inclusion_frequency`, aiming to enhance model performance and adaptability.

### Additional CLI Scripts

1. **`generate-permutations`**: Generates and prints a sorted list of all permutations from a comma separated input. This is provided for the convenience of sweeping over all possible combinations of window sizes and aggregations.

   For example you can directly call **`generate-permutations`** in the command line:

   ```bash
   generate-permutations [2,3,4]
   [2], [2, 3], [2, 3, 4], [2, 4], [3], [3, 4], [4]
   ```

   This could be used in the command line in concert with other calls. For example, the following call:

   ```bash
   meds-tab-xgboost --multirun tabularization.window_sizes=$(generate-permutations [1d,2d,7d,full])
   ```

   would resolve to:

   ```bash
   meds-tab-xgboost --multirun tabularization.window_sizes=[1d],[1d,2d],[1d,2d,7d],[1d,2d,7d,full],[1d,2d,full],[1d,7d],[1d,7d,full],[1d,full],[2d],[2d,7d],[2d,7d,full],[2d,full],[7d],[7d,full],[full]
   ```
