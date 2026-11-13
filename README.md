# House Price Prediction

## Exploratory Data Analysis

- The data contains 1,460 training rows and 1,459 test rows. `SalePrice` is available only in training.
- Missing values account for 6.62% of training cells and 6.75% of test cells. The same six columns exceed 40% missingness in both datasets.
- Overall quality and above-ground living area had the strongest positive correlations with sale price. Applying `log1p` reduced target skewness from 1.88 to 0.12.
- The IQR rule flagged 1,657 cells across 897 training rows. Both datasets had zero duplicate rows and zero constant columns, and all training sale prices were positive.

See the [full EDA report](reports/eda_summary.md) for methods, detailed findings, and planned preprocessing.
