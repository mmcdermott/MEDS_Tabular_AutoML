# Prediction Performance

## XGBoost Model Performance on MIMIC-IV Tasks

Evaluating our tabularization approach for baseline models involved training XGBoost across a spectrum of binary clinical prediction tasks, using data from the MIMIC-IV database. These tasks encompassed diverse outcomes such as mortality predictions over different intervals, readmission predictions, and lengths of stay (LOS) in both ICU and hospital settings.

Each task is characterized by its specific label and prediction time. For instance, predicting "30-day readmission" involves assessing whether a patient returns to the hospital within 30 days, with predictions made at the time of discharge. This allows input features to be derived from the entire duration of the patient's admission. In contrast, tasks like "In ICU Mortality" focus on predicting the occurrence of death using only data from the first 24 or 48 hours of ICU admission. Specifically, we use the terminology "Index Timestamp" to mean the timestamp such that no event included as input will occur later than this point.

We optimize predictive accuracy and model performance by using varied window sizes and aggregations of patient data. This approach allows us to effectively capture and leverage the temporal dynamics and clinical nuances inherent in each prediction task.

### 1. XGBoost Time and Memory Profiling on MIMIC-IV

A single XGBoost run was completed to profile time and memory usage. This was done for each `$TASK` using the following command:

```console
meds-tab-model \
    model_launcher=xgboost \
    "input_dir=${MEDS_RESHARD_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR" \
    "output_model_dir=${OUTPUT_MODEL_DIR}/${TASK}/" "task_name=$TASK"
```

This uses the defaults minimum code inclusion count, window sizes, and aggregations from the [`configs/launch_model.yaml`](https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/main/src/MEDS_tabular_automl/configs/launch_model.yaml) which inherits from the [`configs/tabularization/default.yaml`](https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/main/src/MEDS_tabular_automl/configs/tabularization/default.yaml).

```yaml
min_code_inclusion_count: 10
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
meds-tab-model \
   --multirun \
   model_launcher=xgboost \
   "input_dir=${MEDS_RESHARD_DIR}/data" "output_dir=$OUTPUT_TABULARIZATION_DIR" \
   "output_model_dir=${OUTPUT_MODEL_DIR}/${TASK}/" "task_name=$TASK" \
   "hydra.sweeper.n_trials=1000" "hydra.sweeper.n_jobs=${N_PARALLEL_WORKERS}" \
    tabularization.min_code_inclusion_count=10 \
    tabularization.window_sizes=$(generate-subsets [1d,30d,365d,full]) \
    tabularization.aggs=$(generate-subsets [static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max])
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
  tabularization.min_code_inclusion_count: tag(log, range(10, 1000000))
```

You can override xgboost sweep parameters in the [`configs/model_launcher/xgboost.yaml`](https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/main/src/MEDS_tabular_automl/configs/model_launcher/xgboost.yaml) file.

Note that the XGBoost command shown includes `tabularization.window_sizes` and ` tabularization.aggs` in the parameters to sweep over.

For a complete example on MIMIC-IV and for all of our config files, see the [MIMIC-IV companion repository](https://github.com/mmcdermott/MEDS_TAB_MIMIC_IV).

#### 2.1 XGBoost Performance on MIMIC-IV

| Task                            | Index Timestamp   | AUC   | Minimum Code Inclusion Count | Number of Included Codes\* | Window Sizes           | Aggregations                                                                |
| ------------------------------- | ----------------- | ----- | ---------------------------- | -------------------------- | ---------------------- | --------------------------------------------------------------------------- |
| Post-discharge 30 day Mortality | Discharge         | 0.935 | 1,371                        | 5,712                      | \[7d,full\]            | \[code/count,value/count,value/min,value/max\]                              |
| Post-discharge 1 year Mortality | Discharge         | 0.898 | 289                          | 10,048                     | \[2h,12h,1d,30d,full\] | \[static/present,code/count,value/sum_sqd,value/min\]                       |
| 30 day Readmission              | Discharge         | 0.708 | 303                          | 9,903                      | \[30d,365d,full\]      | \[code/count,value/count,value/sum,value/sum_sqd,value/max\]                |
| In ICU Mortality                | Admission + 24 hr | 0.661 | 7,059                        | 3,037                      | \[12h,full\]           | \[static/present,code/count,value/sum,value/min,value/max\]                 |
| In ICU Mortality                | Admission + 48 hr | 0.673 | 71                           | 16,112                     | \[1d,7d,full\]         | \[static/present,code/count,value/sum,value/min,value/max\]                 |
| In Hospital Mortality           | Admission + 24 hr | 0.812 | 43                           | 18,989                     | \[1d,full\]            | \[static/present,code/count,value/sum,value/min,value/max\]                 |
| In Hospital Mortality           | Admission + 48 hr | 0.810 | 678                          | 7,433                      | \[1d,full\]            | \[static/present,code/count,value/count\]                                   |
| LOS in ICU > 3 days             | Admission + 24 hr | 0.946 | 30,443                       | 1,624                      | \[2h,7d,30d\]          | \[static/present,code/count,value/count,value/sum,value/sum_sqd,value/max\] |
| LOS in ICU > 3 days             | Admission + 48 hr | 0.967 | 2,864                        | 4,332                      | \[2h,7d,30d\]          | \[code/count,value/sum_sqd,value/max\]                                      |
| LOS in Hospital > 3 days        | Admission + 24 hr | 0.943 | 94,633                       | 912                        | \[12h,1d,7d\]          | \[code/count,value/count,value/sum_sqd\]                                    |
| LOS in Hospital > 3 days        | Admission + 48 hr | 0.945 | 30,880                       | 1,619                      | \[1d,7d,30d\]          | \[code/count,value/sum,value/min,value/max\]                                |

- Number of Included Codes is based on Minimum Code Inclusion Count -- we calculated the number of resulting codes that were above the minimum threshold and reported that.

#### 2.2 XGBoost Optimal Found Model Parameters

Additionally, the model parameters from the highest performing run are reported below.

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

For more details about eICU specific task generation and running, see the [eICU companion repository](https://github.com/mmcdermott/MEDS_TAB_EICU).

#### 1. XGBoost Performance on eICU

| Task                            | Index Timestamp   | AUC   | Minimum Code Inclusion Count | Window Sizes             | Aggregations                                                   |
| ------------------------------- | ----------------- | ----- | ---------------------------- | ------------------------ | -------------------------------------------------------------- |
| Post-discharge 30 day Mortality | Discharge         | 0.603 | 68,235                       | \[12h,1d,full\]          | \[code/count,value/sum_sqd,value/max\]                         |
| Post-discharge 1 year Mortality | Discharge         | 0.875 | 3,280                        | \[30d,365d\]             | \[static/present,value/sum,value/sum_sqd,value/min,value/max\] |
| In Hospital Mortality           | Admission + 24 hr | 0.855 | 335,912                      | \[2h,7d,30d,365d,full\]  | \[static/present,code/count,value/count,value/min,value/max\]  |
| In Hospital Mortality           | Admission + 48 hr | 0.570 | 89,121                       | \[12h,1d,30d\]           | \[code/count,value/count,value/min\]                           |
| LOS in ICU > 3 days             | Admission + 24 hr | 0.783 | 7,881                        | \[1d,30d,full\]          | \[static/present,code/count,value/count,value/sum,value/max\]  |
| LOS in ICU > 3 days             | Admission + 48 hr | 0.757 | 1,719                        | \[2h,12h,7d,30d,full\]   | \[code/count,value/count,value/sum,value/sum_sqd,value/min\]   |
| LOS in Hospital > 3 days        | Admission + 24 hr | 0.864 | 160                          | \[1d,30d,365d,full\]     | \[static/present,code/count,value/min,value/max\]              |
| LOS in Hospital > 3 days        | Admission + 48 hr | 0.895 | 975                          | \[12h,1d,30d,365d,full\] | \[code/count,value/count,value/sum,value/sum_sqd\]             |

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
