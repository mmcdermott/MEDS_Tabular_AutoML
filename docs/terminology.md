# Definitions for meds-tab terms

Refer to the terms defined in the [official MEDS Schema](https://github.com/Medical-Event-Data-Standard/meds) and [MEDS_transforms](https://meds-transforms.readthedocs.io/en/latest/terminology/).


## MEDS-Tab Data Types

MEDS Format consists of four core fields:

- `subject_id`: Unique identifier for each patient
- `time`: Timestamp of the measurement (NULL for static data)
- `code`: Feature name/identifier
- `numeric_value`: The measurement value (if applicable)

### Four Types of Data:

!!! note
     "Dynamic" and "time-series" are used interchangeably to describe data that changes over time.

#### 1. Static Codes
- Don't change over time (`time` = NULL)
- Categorical values (no `numeric_value`)
- Examples: gender, blood type, ethnicity

#### 2. Static Numerical Values
- Don't change over time (`time` = NULL)
- Include `numeric_value`
- Examples: birth weight, height at admission

#### 3. Dynamic Codes
- Change over time (`time` required)
- Categorical values (no `numeric_value`)
- Examples: diagnosis codes, medication orders
- Also known as: time-series codes

#### 4. Dynamic Numerical Values
- Change over time (`time` required)
- Include `numeric_value`
- Examples: vital signs, lab results
- Also known as: time-series numerical values
