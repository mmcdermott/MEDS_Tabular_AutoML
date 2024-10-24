# MEDS-Tab Terminology

This document defines key terms used in MEDS-Tab. For complete reference, see the [official MEDS Schema](https://github.com/Medical-Event-Data-Standard/meds) and [MEDS_transforms](https://meds-transforms.readthedocs.io/en/latest/terminology/).

## Core MEDS Fields

| Field           | Definition                                                  |
| --------------- | ----------------------------------------------------------- |
| `subject_id`    | Unique identifier for each patient                          |
| `time`          | Timestamp when the data was recorded (NULL for static data) |
| `code`          | Feature identifier/name                                     |
| `numeric_value` | Measurement value (when applicable)                         |

One example of this is referred to as a `measurement` in MEDS-Tab, which is a single row of data with the fields above. For example:

```yaml
subject_id: patient_123
time: '2024-01-15 14:30:00'
code: HEART_RATE
numeric_value: 72.0
```

represents a heart rate measurement of 72.0 for patient_123 taken on January 15th, 2024 at 2:30 PM.

## Feature Types

Measurements in MEDS-Tab are categorized into four types based on whether they include timestamps and numeric values:

| Term                   | Definition                                                   | Examples                           |
| ---------------------- | ------------------------------------------------------------ | ---------------------------------- |
| Static Codes           | Measurements with no timestamp and no numeric value          | Gender, blood type                 |
| Static Numeric Values  | Measurements with no timestamp but including a numeric value | Birth weight, admission height     |
| Dynamic Codes          | Measurements with a timestamp but no numeric value           | Diagnosis codes, medication orders |
| Dynamic Numeric Values | Measurements with both a timestamp and numeric value         | Vital signs, lab results           |

Note that "Static" and "Dynamic" refer to whether a timestamp is recorded in the MEDS data, not whether the underlying concept can change over time.

## Aggregation Functions

| Aggregation      | Applies To             | Definition                                         |
| ---------------- | ---------------------- | -------------------------------------------------- |
| `static/present` | Static Codes           | Binary indicator of code presence                  |
| `static/first`   | Static Numeric Values  | The numeric value                                  |
| `code/count`     | Dynamic Codes          | Count of code occurrences within lookback window   |
| `value/count`    | Dynamic Numeric Values | Count of measurements within lookback window       |
| `value/sum`      | Dynamic Numeric Values | Sum of measurements within lookback window         |
| `value/sum_sqd`  | Dynamic Numeric Values | Sum of squared measurements within lookback window |
| `value/min`      | Dynamic Numeric Values | Minimum measurement within lookback window         |
| `value/max`      | Dynamic Numeric Values | Maximum measurement within lookback window         |

Static aggregations are computed once per subject_id and static code. Dynamic aggregations are computed per subject_id, code, and lookback window, where the lookback window defines the time period before a reference time point over which measurements are aggregated. Note that the value-based aggregations (`value/*`) are only computed for the subset of dynamic code measurements that include numeric values, while `code/count` is computed for all dynamic codes regardless of whether they have numeric values.

We provide examples of these aggregations here. Notice that for dynamic aggregations, data within a lookback window (e.g., last 24 hours) is input to the aggregation function.

| Aggregation      | Input Data                                 | Result | Explanation                                                           |
| ---------------- | ------------------------------------------ | ------ | --------------------------------------------------------------------- |
| `static/present` | Gender//Female                             | 1      | Indicates the presence (1) of the code "Gender//Female"               |
| `static/first`   | Birth Weight: 3.2 kg                       | 3.2    | Returns the numeric value of the static measurement                   |
| `code/count`     | Heart Rate: \[80, NULL, 78, 90\]           | 4      | Counts the occurrences of codes within the lookback window            |
| `value/count`    | Heart Rate: \[80, 78, 90\]                 | 3      | Counts the number of measurements recorded within the lookback window |
| `value/sum`      | Glucose Levels: \[100, 110, 105\]          | 315    | Sums the measurement values within the lookback window                |
| `value/sum_sqd`  | Blood Pressure Readings: \[120, 125\]      | 30,025 | Sums the squares of the measurements (120² + 125²)                    |
| `value/min`      | Temperature Readings: \[37.5, 38.0, 37.0\] | 37.0   | Finds the minimum value within the lookback window                    |
| `value/max`      | Respiratory Rate: \[16, 18, 20\]           | 20     | Finds the maximum value within the lookback window                    |

## Lookback Window

We define a lookback window as a time period before a reference time point over which dynamic data is aggregated. By default, we use the lookback windows (defined in [this default hydra config](https://github.com/mmcdermott/MEDS_Tabular_AutoML/blob/main/src/MEDS_tabular_automl/configs/tabularization/default.yaml)):

```yaml
window_sizes:
  - 1d   # 1 day
  - 7d   # 7 days
  - 30d   # 30 days
  - 365d   # 1 year
  - full   # full subject history
```
