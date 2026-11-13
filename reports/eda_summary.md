# Exploratory Data Analysis (EDA) — Key Findings

## 1. Dataset Overview

The House Price Prediction project uses two datasets: training and testing.

- **Training dataset:** 1,460 rows and 81 columns.
- **Testing dataset:** 1,459 rows and 80 columns.
- **Target variable:** `SalePrice`, which is present only in the training dataset.

The training dataset contains 38 numerical and 43 categorical columns, while the testing dataset contains 37 numerical and 43 categorical columns. Categorical counts include CSV text columns, which pandas 3 infers as `str`.

`SalePrice` is the dependent (target) variable. The remaining columns represent candidate independent features, except for `Id`, which is a unique identifier and is generally excluded from model training.

## 2. Missing Value Analysis

The training dataset contains **7,829 missing values (6.62%)**, while the testing dataset contains **7,878 missing values (6.75%)**.

- Training dataset: 19 columns contain missing values.
- Testing dataset: 33 columns contain missing values.
- The same six columns have more than 40% missing values in both datasets: `Alley`, `MasVnrType`, `FireplaceQu`, `PoolQC`, `Fence`, and `MiscFeature`.
- The target variable, `SalePrice`, contains no missing values.

Some missing values represent the absence of a property feature, such as a pool, garage, or alley access. These should be handled appropriately during preprocessing rather than automatically removed.

## 3. Numerical Feature Analysis

The training dataset contains 38 numerical columns, including `Id` and `SalePrice`.

The target variable has a skewness of **1.8829**, indicating a strongly right-skewed distribution.

The ten numerical features with the strongest positive Pearson correlations with `SalePrice` are:

| Feature | Correlation |
|---|---:|
| OverallQual | 0.790982 |
| GrLivArea | 0.708624 |
| GarageCars | 0.640409 |
| GarageArea | 0.623431 |
| TotalBsmtSF | 0.613581 |
| 1stFlrSF | 0.605852 |
| FullBath | 0.560664 |
| TotRmsAbvGrd | 0.533723 |
| YearBuilt | 0.522897 |
| YearRemodAdd | 0.507101 |

Overall quality and above-ground living area had the strongest individual linear associations with sale price.

## 4. Categorical Feature Analysis

The training dataset contains 43 categorical columns.

Neighborhood and zoning categories show meaningful differences in average house prices.

- `NAmes` is the most frequent neighborhood, with 225 properties.
- `NoRidge` has the highest average sale price, approximately $335,295.
- `MeadowV` has the lowest average sale price, approximately $98,576.
- `FV` has the highest average sale price among zoning categories, approximately $214,014.
- `C (all)` has the lowest average sale price among zoning categories, approximately $74,528.

These findings suggest that location-related categorical features may be valuable for price prediction.

## 5. Outlier Analysis

The EDA script applies the interquartile range (IQR) rule to 37 numerical training columns. This includes `SalePrice` and excludes `Id`. For each column, it flags values strictly below Q1 − 1.5 × IQR or strictly above Q3 + 1.5 × IQR. Missing cells do not trigger a flag.

The rule flagged **1,657 of 54,020 evaluated cells (3.07%)**, counting missing cells in the denominator as the script does. The flags occurred in 31 columns and covered **897 distinct training rows**. A single property can have several flagged values.

The ten columns with the highest outlier counts are:

| Feature | Outlier count |
|---|---:|
| EnclosedPorch | 208 |
| BsmtFinSF2 | 167 |
| OverallCond | 125 |
| ScreenPorch | 116 |
| MSSubClass | 103 |
| MasVnrArea | 96 |
| LotFrontage | 88 |
| BsmtHalfBath | 82 |
| OpenPorchSF | 77 |
| LotArea | 69 |

Some values flagged by the IQR method may be legitimate, especially in features containing many zeros or highly skewed distributions.

No outliers have been removed during EDA.

## 6. Target Variable Analysis

The `SalePrice` distribution has:

- **Skewness:** 1.8829
- **Excess kurtosis:** 6.5363
- **Skewness after log1p transformation:** 0.1213

The original distribution is strongly right-skewed, with a relatively small number of properties having exceptionally high sale prices.

After applying a temporary `log1p` transformation, the distribution becomes substantially more symmetric.

This suggests that a logarithmic target transformation may be beneficial during model development.

The transformation was used only for EDA and was not permanently applied to the dataset.

## 7. Data Quality Checks

The following checks were completed:

- No duplicate rows were found in either dataset.
- No duplicate `Id` values were found in either dataset.
- `SalePrice` contains no missing values.
- No positive or negative infinite values were detected in numerical columns in either dataset.
- Training and testing feature names match after excluding `SalePrice`.
- The original dataset dimensions were preserved.
- Both datasets have **0 constant columns** using `nunique(dropna=False) == 1`, which counts missingness as a distinct value.
- The training dataset has **0 rows** with `SalePrice <= 0`.

## 8. Recommended Preprocessing Actions

The following actions are recommended based on the EDA findings:

1. Use the data dictionary to distinguish absent property features from unknown values, then choose suitable imputation strategies.
2. Investigate potential outliers before deciding whether to retain, transform, or remove them.
3. Encode categorical features into numerical representations.
4. Create useful engineered features, such as house age and total living area.
5. Consider a logarithmic transformation of `SalePrice`.
6. Apply feature scaling where required by the selected algorithms.
7. Investigate highly correlated and redundant predictors.
8. Exclude `Id` from model predictors.
9. Use a training/validation split and fit preprocessing transformations only on the training portion to prevent data leakage.

**Status** — EDA complete. Preprocessing remains planned.
