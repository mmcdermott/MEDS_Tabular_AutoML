# XGBoost Performance

## D.1 XGBoost Model Performance on MIMIC-IV Across Tasks

Evaluating our tabularization approach for baseline models involved training XGBoost across a spectrum of binary clinical prediction tasks using data from the MIMIC-IV database. These tasks encompassed diverse outcomes such as mortality predictions over different intervals, readmission predictions, and lengths of stay (LOS) in both ICU and hospital settings.

Each task is characterized by its specific label and prediction time. For instance, predicting "30-day readmission" involves assessing whether a patient returns to the hospital within 30 days, with predictions made at the time of discharge. This allows input features to be derived from the entire duration of the patient's admission. In contrast, tasks like "In ICU Mortality" focus on predicting the occurrence of death using only data from the first 24 or 48 hours of ICU admission. Specifically, we use the terminology "Index Timestamp" to mean the timestamp such that no event included as input will occur later than this point.

Optimizing predictive accuracy and model performance necessitated employing varied window sizes and aggregations of patient data. This approach allows us to effectively capture and leverage the temporal dynamics and clinical nuances inherent in each prediction task.

### 1.1 XGBoost Time and Memory Profiling on MIMIC-IV

A single XGBoost run was completed to profile time and memory usage. This was done for each `$TASK` using the following command:

```
meds-tab-xgboost
      MEDS_cohort_dir="path_to_data" \
      task_name=$TASK \
      output_dir="output_directory" \
      do_overwrite=False \
```

This uses the defaults minimum code inclusion frequency, window sizes, and aggregations from the `launch_xgboost.yaml`:

```yaml
allowed_codes:      # allows all codes that meet min code inclusion frequency
min_code_inclusion_frequency: 10
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

Since this includes every window size and aggregation, it is the most expoensive to run. The runtimes and memory usage are reported below.

#### 1.1.1 XGBoost Runtimes and Memory Usage on MIMIC-IV Tasks

| Task                            | Index Timestamp   | Real Time | User Time | Sys Time  | Avg Memory (MiB) | Peak Memory (MiB) |
| ------------------------------- | ----------------- | --------- | --------- | --------- | ---------------- | ----------------- |
| Post-discharge 30 day Mortality | Discharge         | 2m59.807s | 3m38.285s | 0m38.274s | 9036.728         | 11955.312         |
| Post-discharge 1 year Mortality | Discharge         | 5m16.958s | 6m10.065s | 0m58.964s | 10803.670        | 12330.355         |
| 30 day Readmission              | Discharge         | 2m30.609s | 3m3.836s  | 0m39.415s | 13198.832        | 18676.605         |
| In ICU Mortality                | Admission + 24 hr | 0m38.275s | 1m3.656s  | 0m13.198s | 1711.942         | 2985.699          |
| In ICU Mortality                | Admission + 48 hr | 0m34.669s | 1m1.389s  | 0m12.742s | 1613.256         | 2770.438          |
| In Hospital Mortality           | Admission + 24 hr | 2m8.912s  | 2m41.180s | 0m31.707s | 9071.615         | 12055.980         |
| In Hospital Mortality           | Admission + 48 hr | 1m54.025s | 2m25.322s | 0m28.925s | 8857.783         | 12370.898         |
| LOS in ICU > 3 days             | Admission + 24 hr | 2m2.689s  | 2m37.196s | 0m28.110s | 4650.008         | 5715.484          |
| LOS in ICU > 3 days             | Admission + 48 hr | 1m44.434s | 2m18.558s | 0m23.825s | 4453.363         | 5577.164          |
| LOS in Hospital > 3 days        | Admission + 24 hr | 6m4.884s  | 7m5.025s  | 1m4.335s  | 11011.710        | 12223.449         |
| LOS in Hospital > 3 days        | Admission + 48 hr | 6m9.587s  | 7m12.853s | 1m3.858s  | 10703.064        | 11829.742         |

#### 1.1.2 MIMIC-IV Task Specific Training Cohort Size

To better understand the runtimes, we also report the task specific cohort size.

| Task                            | Index Timestamp   | Number of Patients | Number of Events |
| ------------------------------- | ----------------- | ------------------ | ---------------- |
| Post-discharge 30 day Mortality | Discharge         | 149014             | 356398           |
| Post-discharge 1 year Mortality | Discharge         | 149014             | 356398           |
| 30 day Readmission              | Discharge         | 17418              | 377785           |
| In ICU Mortality                | Admission + 24 hr | 7839               | 22811            |
| In ICU Mortality                | Admission + 48 hr | 6750               | 20802            |
| In Hospital Mortality           | Admission + 24 hr | 51340              | 338614           |
| In Hospital Mortality           | Admission + 48 hr | 47231              | 348289           |
| LOS in ICU > 3 days             | Admission + 24 hr | 42809              | 61342            |
| LOS in ICU > 3 days             | Admission + 48 hr | 42805              | 61327            |
| LOS in Hospital > 3 days        | Admission + 24 hr | 152126             | 360208           |
| LOS in Hospital > 3 days        | Admission + 48 hr | 152120             | 359020           |

### 1.2 MIMIC-IV Sweep

The XGBoost sweep was run using the following command for each `$TASK`:

```
meds-tab-xgboost --multirun \
      MEDS_cohort_dir="path_to_data" \
      task_name=$TASK \
      output_dir="output_directory" \
      tabularization.window_sizes=$(generate-permutations [1d,30d,365d,full]) \
      do_overwrite=False \
      tabularization.aggs=$(generate-permutations [static/present,code/count,value/count,value/sum,value/sum_sqd,value/min,value/max])
```

The model parameters were set to:

```yaml
model:
  booster: gbtree
  device: cpu
  nthread: 1
  tree_method: hist
  objective: binary:logistic
```

The hydra sweeper swept over the parameters:

```yaml
params:
  +model_params.model.eta: tag(log, interval(0.001, 1))
  +model_params.model.lambda: tag(log, interval(0.001, 1))
  +model_params.model.alpha: tag(log, interval(0.001, 1))
  +model_params.model.subsample: interval(0.5, 1)
  +model_params.model.min_child_weight: interval(1e-2, 100)
  +model_params.model.max_depth: range(2, 16)
  model_params.num_boost_round: range(100, 1000)
  model_params.early_stopping_rounds: range(1, 10)
  tabularization.min_code_inclusion_frequency: tag(log, range(10, 1000000))
```

Note that the XGBoost command shown includes `tabularization.window_sizes` and ` tabularization.aggs` in the parameters to sweep over.

For a complete example on MIMIC-IV and for all of our config files, see the [MIMIC-IV companion repository](https://github.com/mmcdermott/MEDS_TAB_MIMIC_IV).

#### 1.2.1 XGBoost Performance on MIMIC-IV

| Task                            | Index Timestamp   | AUC          | Minimum Code Inclusion Frequency | Number of Included Codes\* | Window Sizes           | Aggregations                                                                |
| ------------------------------- | ----------------- | ------------ | -------------------------------- | -------------------------- | ---------------------- | --------------------------------------------------------------------------- |
| Post-discharge 30 day Mortality | Discharge         | 0.9347633541 | 1371                             | 5712                       | \[7d,full\]            | \[code/count,value/count,value/min,value/max\]                              |
| Post-discharge 1 year Mortality | Discharge         | 0.8979986449 | 289                              | 10048                      | \[2h,12h,1d,30d,full\] | \[static/present,code/count,value/sum_sqd,value/min\]                       |
| 30 day Readmission              | Discharge         | 0.7076685566 | 303                              | 9903                       | \[30d,365d,full\]      | \[code/count,value/count,value/sum,value/sum_sqd,value/max\]                |
| In ICU Mortality                | Admission + 24 hr | 0.6612338171 | 7059                             | 3037                       | \[12h,full\]           | \[static/present,code/count,value/sum,value/min,value/max\]                 |
| In ICU Mortality                | Admission + 48 hr | 0.671986067  | 71                               | 16112                      | \[1d,7d,full\]         | \[static/present,code/count,value/sum,value/min,value/max\]                 |
| In Hospital Mortality           | Admission + 24 hr | 0.8119187166 | 43                               | 18989                      | \[1d,full\]            | \[static/present,code/count,value/sum,value/min,value/max\]                 |
| In Hospital Mortality           | Admission + 48 hr | 0.8100362943 | 678                              | 7433                       | \[1d,full\]            | \[static/present,code/count,value/count\]                                   |
| LOS in ICU > 3 days             | Admission + 24 hr | 0.9455099633 | 30443                            | 1624                       | \[2h,7d,30d\]          | \[static/present,code/count,value/count,value/sum,value/sum_sqd,value/max\] |
| LOS in ICU > 3 days             | Admission + 48 hr | 0.9667108784 | 2864                             | 4332                       | \[2h,7d,30d\]          | \[code/count,value/sum_sqd,value/max\]                                      |
| LOS in Hospital > 3 days        | Admission + 24 hr | 0.9434966078 | 94633                            | 912                        | \[12h,1d,7d\]          | \[code/count,value/count,value/sum_sqd\]                                    |
| LOS in Hospital > 3 days        | Admission + 48 hr | 0.9449770561 | 30880                            | 1619                       | \[1d,7d,30d\]          | \[code/count,value/sum,value/min,value/max\]                                |

- Number of Included Codes is based on Minimum Code Inclusion Frequency -- we calculated the number of resulting codes that were above the minimum threshold and reported that.

#### 1.2.2 XGBoost Optimal Found Model Parameters

Additionally, the model parameters from the highest performing run are reported below.

| Task                            | Index Timestamp   | Eta            | Lambda         | Alpha         | Subsample    | Minimum Child Weight | Number of Boosting Rounds | Early Stopping Rounds | Max Tree Depth |
| ------------------------------- | ----------------- | -------------- | -------------- | ------------- | ------------ | -------------------- | ------------------------- | --------------------- | -------------- |
| Post-discharge 30 day Mortality | Discharge         | 0.005630897092 | 0.03218837176  | 0.3741846464  | 0.5716492359 | 52.66844896          | 703                       | 9                     | 16             |
| Post-discharge 1 year Mortality | Discharge         | 0.008978198787 | 0.086075240914 | 0.342564218   | 0.8994363088 | 75.94359197          | 858                       | 9                     | 11             |
| 30 day Readmission              | Discharge         | 0.005970244514 | 0.3591376982   | 0.3741846464  | 0.673450045  | 52.66844896          | 712                       | 9                     | 16             |
| In ICU Mortality                | Admission + 24 hr | 0.03824348927  | 0.06183970736  | 0.2310791064  | 0.9947482627 | 88.53086045          | 513                       | 7                     | 14             |
| In ICU Mortality (first 48h)    | Admission + 48 hr | 0.04373178504  | 0.04100575186  | 0.2888938852  | 0.9617417624 | 90.881739            | 484                       | 5                     | 14             |
| In Hospital Mortality           | Admission + 24 hr | 0.02790651024  | 0.01319397229  | 0.0105408763  | 0.5673852112 | 11.22281297          | 454                       | 6                     | 9              |
| In Hospital Mortality           | Admission + 48 hr | 0.01076063059  | 0.06007544254  | 0.1791900222  | 0.9641152835 | 83.69584368          | 631                       | 7                     | 13             |
| LOS in ICU > 3 days             | Admission + 24 hr | 0.01203878234  | 0.08963582145  | 0.1367180869  | 0.6264012852 | 26.20493325          | 650                       | 8                     | 14             |
| LOS in ICU > 3 days             | Admission + 48 hr | 0.01203878234  | 0.04882102808  | 0.1997059646  | 0.9608288859 | 83.9736355           | 615                       | 7                     | 13             |
| LOS in Hospital > 3 days        | Admission + 24 hr | 0.008389745342 | 0.06656965098  | 0.2553069741  | 0.9886841026 | 89.89987526          | 526                       | 5                     | 14             |
| LOS in Hospital > 3 days        | Admission + 48 hr | 0.00121145622  | 0.03018152667  | 0.02812771908 | 0.9671829656 | 8.657613623          | 538                       | 8                     | 7              |
