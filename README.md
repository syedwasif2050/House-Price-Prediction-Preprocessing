# House Price Prediction: Data Preprocessing Pipeline
A comprehensive data engineering and preprocessing application built in Python using the `Pandas` library. This script establishes a robust pipeline to clean and prepare raw housing datasets for predictive Machine Learning models.

##  Features & Pipeline Stages
- **Data Profiling & Exploration:** Evaluates dataset dimensions, handles column data types, and extracts summary statistics (`describe`).
- **Automated Missing Value Imputation:** - Imputes numerical missing features using the **Mean** value of the column.
  - Imputes categorical text missing features using the **Mode** (most frequent value).
- **Deduplication:** Automatically scans and removes duplicate data entries to maintain data integrity.
- **Feature Selection:** Drops non-predictive features like `Id` columns to optimize feature spaces.
- **Categorical Encoding:** Converts textual categories into machine-readable numerical formats using **One-Hot Encoding** (`pd.get_dummies`).
