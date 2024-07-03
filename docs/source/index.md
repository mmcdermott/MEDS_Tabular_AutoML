# Welcome!

MEDS-Tab is a library designed for automated tabularization, data preparation with aggregation, and time windowing. Check out below for an overview of MEDS-Tab and how it could be useful in your workflows!

```{toctree}
---
glob:
maxdepth: 2
---
Overview <overview>
Pipeline <implementation>
Memory/CPU Usage <profiling>
Performance <prediction>
API <api/modules>
```

______________________________________________________________________

## Why MEDS-Tab?

MEDS-Tab is a comprehensive framework designed to streamline the handling, modeling, and analysis of complex medical time-series data. By leveraging automated processes, MEDS-Tab significantly reduces the computation required to generate high-quality baseline models for diverse supervised learning tasks.

- Cost Efficiency: MEDS-Tab is dramatically more cost-effective compared to existing solutions
- Strong Performance: MEDS-Tab provides robustness and high performance across various datasets compared with other frameworks.

### I. Transform to MEDS

MEDS-Tab leverages the recently developed, minimal, easy-to-use Medical Event Data Standard (MEDS) schema to standardize structured EHR data to a consistent schema from which baselines can be reliably produced across arbitrary tasks and settings. In order to use MEDS-Tab, you will first need to transform your raw EHR data to a MEDS format, which can be done using the following libraries:

- [MEDS Polars](https://github.com/mmcdermott/MEDS_polars_functions) for a set of functions and scripts for extraction to and transformation/pre-processing of MEDS-formatted data.
- [MEDS ETL](https://github.com/Medical-Event-Data-Standard/meds_etl) for a collection of ETLs from common data formats to MEDS. The package library currently supports MIMIC-IV, OMOP v5, and MEDS FLAT (a flat version of MEDS).

### II. Run MEDS-Tab

- Run the MEDS-Tab Command-Line Interface tool (`MEDS-Tab-cli`) to extract cohorts based on your task - check out the [Usage Guide](https://meds-tab--36.org.readthedocs.build/en/36/overview.html#core-cli-scripts-overview)!

- Painless Reproducibility: Use [MEDS-Tab](https://github.com/mmcdermott/MEDS_TAB_MIMIC_IV/tree/main/tasks) to obtain comparable, reproducible, and well-tuned XGBoost results tailored to your dataset-specific feature space!

By following these steps, you can seamlessly transform your dataset, define necessary criteria, and leverage powerful machine learning tools within the MEDS-Tab ecosystem. This approach not only simplifies the process but also ensures high-quality, reproducible results for your machine learning tasks for health projects. It can reliably take no more than a week of full-time human effort to perform Steps I-V on new datasets in reasonable raw formulations!
