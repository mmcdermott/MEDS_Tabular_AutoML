# Computational Performance vs. Existing Pipelines

Evaluating the computational overhead of tabularization methods is essential for assessing their efficiency and suitability for large-scale medical data processing. This section presents a comparative analysis of the computational overhead of MEDS-Tab with other systems like Catabra and TSFresh. It outlines the performance of each system in terms of wall time, memory usage, and output size, highlighting the computational efficiency and scalability of MEDS-Tab.

## 1. System Comparison Overview

The systems compared in this study represent different approaches to data tabularization, with the main difference being MEDS-Tab usage of sparse tabularization. Specifically, for comparison we used:

1. **Catabra/Catabra-Mem**: Offers data processing capabilities for time-series medical data, with variations to test memory management.
2. **TSFresh**: Both known and used for extensive feature extraction capabilities.

The benchmarking tests were conducted using the following hardware and software settings:

- **CPU Specification**: 2 x AMD EPYC 7713 64-Core Processor
- **RAM Specification**: 1024GB, 3200MHz, DDR4
- **Software Environment**: Ubuntu 22.04.4 LTS

### MEDS-Tab Tabularization Technique

Tabularization of time-series data, as depecited above, is commonly used in several past works. The only two libraries to our knowledge that provide a full tabularization pipeline are `tsfresh` and `catabra`. `catabra` also offers a slower but more memory efficient version of their method which we denote `catabra-mem`. Other libraries either provide only rolling window functionalities (`featuretools`) or just pivoting operations (`Temporai`/`Clairvoyance`, `sktime`, `AutoTS`). We provide a significantly faster and more memory efficient method. Our findings show that on the MIMIC-IV and eICU medical datasets we significantly outperform both above-mentioned methods that provide similar functionalities with MEDS-Tab. While `catabra` and `tsfresh` could not even run within a budget of 10 minutes on as low as 10 patient's data for eICU, our method scales to process hundreds of patients with low memory usage under the same time budget. We present the results below.

## 2. Comparative Performance Analysis

The tables below detail computational resource utilization across two datasets and various patient scales, emphasizing the better performance of MEDS-Tab in all of the scenarios. The tables are organized by dataset and number of patients. For the analysis, the full window sizes and the aggregation method code_count were used. Additionally, we use a budget of 10 minutes for running our tests given that for such small number of patients (10, 100, and 500 patients) data should be processed quickly. Note that `catabra-mem` is omitted from the tables as it never completed within the 10 minute budget.

### eICU Dataset

The only method that was able to tabularize eICU data was MEDS-Tab. We ran our method with both 100 and 500 patients, resulting in an increment by three times in the number of codes. MEDS-Tab gave efficient results in terms of both time and memory usage.

a) 100 Patients

**Table 1: 6,374 Codes, 2,065,608 Rows, Output Shape \[132,461, 6,374\]**

| Wall Time | Avg Memory | Peak Memory | Output Size | Method   |
| --------- | ---------- | ----------- | ----------- | -------- |
| 0m39s     | 5,271 MB   | 14,791 MB   | 362 MB      | meds_tab |

b) 500 Patients

**Table 2: 18,314 Codes, 8,737,355 Rows, Output Shape \[565,014, 18,314\]**

| Wall Time | Avg Memory | Peak Memory | Output Size | Method   |
| --------- | ---------- | ----------- | ----------- | -------- |
| 3m4s      | 8,335 MB   | 15,102 MB   | 1,326 MB    | meds_tab |

### MIMIC-IV Dataset

MEDS-Tab, `tsfresh`, and `catabra` were tested across three different patient scales on MIMIC-IV.

a) 10 Patients

This table illustrates the efficiency of MEDS-Tab in processing a small subset of patients with extremely low computational cost and high data throughput, outperforming `tsfresh` and `catabra` in terms of both time and memory efficiency.

**Table 3: 1,504 Codes, 23,346 Rows, Output Shape \[2,127, 1,504\]**

| Wall Time | Avg Memory | Peak Memory | Output Size | Method   |
| --------- | ---------- | ----------- | ----------- | -------- |
| 0m2s      | 423 MB     | 943 MB      | 7 MB        | meds_tab |
| 1m41s     | 84,159 MB  | 265,877 MB  | 1 MB        | tsfresh  |
| 0m15s     | 2,537 MB   | 4,781 MB    | 1 MB        | catabra  |

b) 100 Patients

The performance gap was further highlighted with an increased number of patients and codes. For a moderate patient count, MEDS-Tab demonstrated superior performance with significantly lower wall times and memory usage compared to `tsfresh` and `catabra`.

**Table 4: 4,154 Codes, 150,789 Rows, Output Shape \[15,664, 4,154\]**

| Wall Time | Avg Memory | Peak Memory | Output Size | Method   |
| --------- | ---------- | ----------- | ----------- | -------- |
| 0m5s      | 718 MB     | 1,167 MB    | 45 MB       | meds_tab |
| 5m9s      | 217,477 MB | 659,735 MB  | 4 MB        | tsfresh  |
| 3m17s     | 14,319 MB  | 28,342 MB   | 4 MB        | catabra  |

c) 500 Patients

Scaling further to 500 patients, MEDS-Tab maintained consistent performance, reinforcing its capability to manage large datasets efficiently. Because of the set time limit of 10 minutes, we could not get results for `catabra` and `tsfresh`. In comparison, MEDS-Tab processed the data in about 15 seconds, making it at least 40 times faster for the given patient scale.

**Table 5: 48,115 Codes, 795,368 Rows, Output Shape \[75,595, 8,115\]**

| Wall Time | Avg Memory | Peak Memory | Output Size | Method   |
| --------- | ---------- | ----------- | ----------- | -------- |
| 0m16s     | 1,410 MB   | 3,539 MB    | 442 MB      | meds_tab |

______________________________________________________________________
