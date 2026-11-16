# Data Cleaning Summary

## Dataset Split

The labeled data was split into 80% training (1,168 rows) and 20% validation (292 rows) using `random_state=42`. `SalePrice` was used as the target, and `Id` was excluded from the features. The 1,459-row test set was kept separate from the split and cleaned for prediction. Test IDs were saved separately.

## Missing Values

Missing categorical values were filled with `Unknown`. Missing numerical values in the training, validation, and test sets were filled with the corresponding column medians calculated from `X_train` only. This prevents validation and test data from influencing the imputation statistics.

All three cleaned feature datasets contain 79 columns and zero missing values.

## Saved Files

| Data | File in `data/processed/` | Rows |
|---|---|---:|
| Training features | `X_train_clean.csv` | 1,168 |
| Validation features | `X_val_clean.csv` | 292 |
| Test features | `X_test_clean.csv` | 1,459 |
| Training target | `y_train.csv` | 1,168 |
| Validation target | `y_val.csv` | 292 |
| Test IDs | `test_ids.csv` | 1,459 |