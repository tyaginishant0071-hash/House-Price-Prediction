# House Price Prediction

## Exploratory Data Analysis

- The data contains 1,460 training rows and 1,459 test rows. `SalePrice` is available only in training.
- Missing values account for 6.62% of training cells and 6.75% of test cells. The same six columns exceed 40% missingness in both datasets.
- Overall quality and above-ground living area had the strongest positive correlations with sale price. Applying `log1p` reduced target skewness from 1.88 to 0.12.
- The IQR rule flagged 1,657 cells across 897 training rows. Both datasets had zero duplicate rows and zero constant columns, and all training sale prices were positive.

See the [full EDA report](reports/eda_summary.md) for methods, detailed findings, and planned preprocessing.

## Data Cleaning

- Split the labeled data into 80% training (1,168 rows) and 20% validation (292 rows) using `random_state=42`. The 1,459 test rows were cleaned separately.
- Filled missing categorical values with `Unknown` and used medians calculated from the training split to fill missing numerical values in all three sets.
- Verified that each cleaned feature set has 79 columns and zero missing values. Saved the cleaned features, training and validation targets, and test IDs in `data/processed/`.

See the [full data cleaning report](reports/data_cleaning_summary.md) for details and the list of saved files.

## Feature Engineering

- Added five numerical features across the training, validation, and test sets: `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, and `TotalPorchSF`.
- Set four negative engineered age values to zero. All three datasets have the same 84 columns in the same order, with zero missing values and zero negative engineered ages.
- Saved the engineered feature sets as separate CSVs in `data/processed/`.

See the [full feature engineering report](reports/feature_engineering_summary.md) for formulas, verification results, and output filenames.

## Data Preprocessing

- One-hot encoded 43 categorical features into 265 numerical columns and scaled the 41 original numerical features using `StandardScaler`. Both transformers were fitted on training data only and applied to training, validation, and test data.
- The final datasets contain 306 numerical features with identical column order, zero missing values, and zero duplicate column names. Row counts are 1,168 for training, 292 for validation, and 1,459 for test.
- Saved the preprocessed CSVs in `data/processed/` and the fitted transformers and feature-column lists in `models/`. The datasets are ready for model training and evaluation.

See the [full data preprocessing report](reports/data_preprocessing_summary.md) for details and saved filenames.
