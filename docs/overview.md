# The MEDS-Tab Architecture

MEDS-Tab addresses two key challenges in healthcare machine learning: efficiently tabularizing large-scale electronic health record (EHR) data and training competitive baseline models on this tabularized data. This document outlines the architecture and implementation details of MEDS-Tab's pipeline.

MEDS-Tab is designed to scale to hundreds of millions of events and tens of thousands of unique medical codes. Performance optimization is achieved through:

* Efficient parallel processing when appropriate
* Strategic use of sparse data structures
* Memory-aware data loading and processing
* Configurable processing parameters for different hardware capabilities

## Overview

The MEDS-Tab pipeline consists of six main stages, with the first being optional. The pipeline begins with an optional (1) data resharding stage that optimizes processing by restructuring input data into manageable chunks. This is followed by (2) data description, which computes some summary statistics over the features in the dataset. The core processing happens in the (3) static and (4) time-series tabularization stages, which transform the data into a format suitable for tabular machine learning. (5) Task-specific data caching then aligns this data with prediction tasks, and finally, the (6) model training stage provides efficient training capabilities with support for multiple model types and hyperparameter optimization.

## Memory Management Via Sparse Data Structures

Memory management is a central consideration in MEDS-Tab's design. The system employs several key strategies to handle large-scale medical datasets efficiently:

Sparse matrix operations form the foundation of our memory management approach. We utilize scipy.sparse for memory-efficient storage of sparse non-zero elements, which is particularly effective for medical data where most potential features are not present for any given patient at any given time.

Data sharding complements our sparse matrix approach by breaking data into manageable chunks. This enables both memory-efficient processing and parallelization. Shards are processed independently, allowing us to handle datasets that would be impossible to process as a single unit.

The system implements efficient aggregation using Polars for fast rolling window computations. This optimizes same-day event aggregation and maintains memory efficiency during temporal calculations.

## Improved Computational Speed Via Parallel Processing

Our processing strategy differentiates between sequential and parallel operations based on computational needs and data dependencies. The data description and static tabularization stages operate sequentially, as they have manageable computational requirements. In contrast, time-series tabularization, task-specific caching, and model training leverage parallel processing over independent workers (which may be spawned on different cores on a local machine or over a slurm cluster) to handle their more intensive computational demands.

Data flow through the pipeline is optimized through caching and sharding. Each stage's output is structured to minimize memory requirements while maintaining accessibility for subsequent stages. The system preserves sparsity wherever possible and uses efficient shard management to increase processing speed and reduce total memory consumption.

## Feature Engineering Via Rolling Windows and Aggregation Functions

MEDS-Tab implements a comprehensive feature engineering approach that handles both static and temporal data. For static features, we capture both presence and first-recorded values (as there should be only one occurrence of a static code). Time-series features are processed through various aggregation methods including counts, sums, minimums, and maximums. These aggregations can be computed over multiple time windows (1 day, 30 days, 365 days, or the full patient history), providing temporal context at different scales.

Our feature engineering framework maintains flexibility while enforcing consistency. All aggregations preserve sparsity where possible, and the system includes configurable thresholds for feature inclusion based on frequency and relevance to the target task.

## Model Support and Normalization/Imputation Options

The architecture includes robust support for multiple model types, with XGBoost as the primary implementation. Additional supported models include KNN Classifier, Logistic Regression, Random Forest Classifier, and SGD Classifier. An experimental AutoGluon integration provides automated model selection and tuning capabilities.

Data processing options are designed to maintain efficiency while providing necessary transformations. Normalization options (standard scaler, max abs scaler) preserve sparsity, while imputation methods (mean, median, mode) are available when dense representations are required or beneficial.

## Additional Design Considerations

**Extensibility and maintainability**: The pipeline's modular design allows for the addition of new feature types, aggregation methods, and model support. Contributions are welcome!

**Highly Configurable**: This pipeline is highly configurable via parameters that allow users to adjust processing based on their specific needs and hardware constraints. See the usage guide for more details.
