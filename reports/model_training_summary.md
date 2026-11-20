# Model Training and Evaluation

## 1. Data Loading

Loaded the cleaned and preprocessed training and validation datasets from `data/processed/`.

Dataset shapes:

| Dataset | Shape       |
| ------- | ----------- |
| X_train | (1168, 306) |
| X_val   | (292, 306)  |
| y_train | (1168, 1)   |
| y_val   | (292, 1)    |

The models were trained using the training dataset and evaluated using the validation dataset.

## 2. Model Training

Three regression models were trained to predict house prices.

### Linear Regression

Linear Regression was used as the baseline model to evaluate initial prediction performance.

### Random Forest Regressor

Random Forest Regressor was trained with the following parameters:

- `n_estimators=100`
- `random_state=42`

### Gradient Boosting Regressor

Gradient Boosting Regressor was trained with:

- `n_estimators=100`
- `random_state=42`

## 3. Model Evaluation

All three models were evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.

| Model             | MAE           | RMSE          | R²         |
| ----------------- | ------------- | ------------- | ---------- |
| Linear Regression | 21,216.05     | 65,621.43     | 0.4386     |
| Random Forest     | 17,938.73     | 29,827.31     | 0.8840     |
| Gradient Boosting | **16,360.67** | **26,772.24** | **0.9066** |

## 4. Best Model Selection

Gradient Boosting Regressor achieved the lowest MAE and RMSE and the highest R² score among the three models.

Therefore, Gradient Boosting was selected as the best baseline model for further hyperparameter tuning.

## 5. Conclusion

Three machine learning regression models were successfully trained and evaluated.

Gradient Boosting achieved the best validation performance and was selected for the Model Optimization phase.
