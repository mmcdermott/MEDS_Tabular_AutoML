# MEDS-Tab Terminology

This document defines key terms used in MEDS-Tab. For complete reference, see the [official MEDS Schema](https://github.com/Medical-Event-Data-Standard/meds) and [MEDS_transforms](https://meds-transforms.readthedocs.io/en/latest/terminology/).

## Core MEDS Fields

| Field | Definition |
|-------|------------|
| `subject_id` | Unique identifier for each patient |
| `time` | Timestamp when the data was recorded (NULL for static data) |
| `code` | Feature identifier/name |
| `numeric_value` | Measurement value (when applicable) |

One example of this is referred to as a `measurement` in MEDS-Tab, which is a single row of data with the fields above. For example:
```yaml
subject_id: "patient_123"
time: "2024-01-15 14:30:00"
code: "HEART_RATE"
numeric_value: 72.0
```

## Feature Types

Measurements in MEDS-Tab are categorized into four types based on whether they include timestamps and numeric values:

| Term | Definition | Examples |
|------|------------|-----------|
| Static Codes | Measurements with no timestamp and no numeric value | Gender, blood type |
| Static Numeric Values | Measurements with no timestamp but including a numeric value | Birth weight, admission height |
| Dynamic Codes | Measurements with a timestamp but no numeric value | Diagnosis codes, medication orders |
| Dynamic Numeric Values | Measurements with both a timestamp and numeric value | Vital signs, lab results |

Note: "Static" and "dynamic" refer to how data is recorded in the dataset structure, not whether the underlying concept can change over time.

## Aggregation Functions

### Static Aggregations
Terms for aggregations applied to static data per subject_id and static code:
- `static/present`: Binary indicator of code presence
- `static/first`: The numeric value (for static numeric values)

### Dynamic Aggregations
Terms for aggregations applied per subject_id and lookback window:

For dynamic codes:
- `code/count`: Count of code occurrences

For dynamic numeric values (subset of dynamic codes with numeric measurements):
- `value/count`: Count of measurements
- `value/sum`: Sum of measurements
- `value/sum_sqd`: Sum of squared measurements
- `value/min`: Minimum measurement
- `value/max`: Maximum measurement

## Lookback Window

A time period before a reference time point over which dynamic data is aggregated. Standard windows:
- "1d": 1 day
- "7d": 7 days
- "30d": 30 days
- "365d": 1 year
- "full": Complete subject history
