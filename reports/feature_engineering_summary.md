# Feature Engineering Summary

## 1. Data Loading and Inspection

Loaded the cleaned training, validation, and test datasets from `data/processed/`. The script checked their shapes and missing values, identified numerical and categorical columns in the training data, and applied the same Python function to all three datasets.

## 2. New Features Created

| Feature | Formula |
| --- | --- |
| `TotalSF` | `TotalBsmtSF + GrLivArea` |
| `TotalBath` | `FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath` |
| `HouseAge` | `YrSold - YearBuilt` |
| `RemodAge` | `YrSold - YearRemodAdd` |
| `TotalPorchSF` | `OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch` |

## 3. Data Verification

Recalculating the age features from the cleaned inputs showed four negative values before adjustment. Training had one negative `RemodAge`, validation had none, and test had one negative `HouseAge` and two negative `RemodAge` values. The feature engineering function set each of these values to zero.

The saved datasets have the same columns in the same order, zero missing values, and zero negative values in either age column.

## 4. Final Dataset Shapes

| Dataset | Rows | Columns |
| --- | ---: | ---: |
| Training | 1,168 | 84 |
| Validation | 292 | 84 |
| Test | 1,459 | 84 |

## 5. Saved Datasets

The feature-engineered datasets were saved separately from the cleaned inputs in `data/processed/`.

- `X_train_fe.csv`
- `X_val_fe.csv`
- `X_test_fe.csv`

## 6. Conclusion

The five engineered numerical features were verified across all three datasets. The saved datasets are ready for the next preprocessing stage.
