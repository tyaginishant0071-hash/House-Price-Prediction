# Data Preprocessing Summary

## 1. Data Loading and Inspection

Loaded the feature-engineered datasets from `data/processed/`.

Verified that all three datasets contain zero missing values.

| Dataset     | Shape      |
| ----------- | ---------- |
| train_fe_df | (1168, 84) |
| val_fe_df   | (292, 84)  |
| test_fe_df  | (1459, 84) |

## 2. Numerical and Categorical Features

Separated the 84 features into:

- Numerical features: 41
- Categorical features: 43

Total features: 41 + 43 = 84.

## 3. One-Hot Encoding

Imported `OneHotEncoder` from `sklearn.preprocessing`.

Initialized the encoder with:

- `handle_unknown='ignore'` to handle unseen categories.
- `sparse_output=False` to obtain dense output.
- Pandas output enabled to preserve encoded column names.

Fitted the encoder only on the 43 categorical columns of the training dataset.

Created a reusable function to transform the categorical columns of the training, validation, and test datasets.

### Encoded Dataset Shapes

| Dataset          | Shape       |
| ---------------- | ----------- |
| train_fe_encoded | (1168, 265) |
| val_fe_encoded   | (292, 265)  |
| test_fe_encoded  | (1459, 265) |

The 43 categorical features were converted into 265 numerical features.

Also combined the original 41 numerical features with the 265 encoded features to create intermediate datasets containing 306 columns.

## 4. Numerical Feature Scaling

Imported `StandardScaler` from `sklearn.preprocessing`.

Created a scaler object named `st_scaler`.

Fitted the scaler only on the 41 numerical columns of the training dataset.

Created a reusable function to transform the numerical columns in all three datasets using the fitted scaler.

Preserved the original column names and DataFrame indexes.

Combined the 41 scaled numerical features with the 265 one-hot encoded features.

## 5. Final Preprocessed Datasets

| Dataset            | Shape       |
| ------------------ | ----------- |
| train_preprocessed | (1168, 306) |
| val_preprocessed   | (292, 306)  |
| test_preprocessed  | (1459, 306) |

### Final Verification

- All three datasets have identical feature columns in the same order.
- All 306 columns are numerical.
- Missing values: 0.
- Duplicate column names: 0.
- Original row counts were preserved.

Both preprocessing transformers were fitted using training data only to prevent data leakage.

## 6. Saved Files

Saved the final preprocessed datasets inside `data/processed/`:

- `X_train_preprocessed.csv`
- `X_val_preprocessed.csv`
- `X_test_preprocessed.csv`

Saved the fitted preprocessing objects and feature-column names inside `models/` using `joblib`:

- `one_hot_encoder.joblib`
- `standard_scaler.joblib`
- `numerical_columns.joblib`
- `categorical_columns.joblib`

## 7. Conclusion

Data preprocessing was completed successfully.

The datasets contain 306 numerical features and are ready for the Model Training and Evaluation phase.