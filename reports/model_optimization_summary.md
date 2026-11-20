# Hyperparameter Tuning and Model Optimization

## 1. Objective

Hyperparameter tuning was performed to improve the performance of the Gradient Boosting Regressor, which was selected as the best model during the Model Training phase.

The objective was to reduce prediction errors and improve house-price prediction accuracy.

## 2. Hyperparameter Tuning

GridSearchCV was used with 5-fold cross-validation (`cv=5`) to identify the best hyperparameter combination.

The training dataset was divided into five folds. During each iteration, four folds were used for training and one fold for validation.

### Parameter Grid

| Hyperparameter | Values Tested |
| -------------- | ------------- |
| n_estimators   | 100, 200      |
| learning_rate  | 0.05, 0.1     |
| max_depth      | 2, 3          |

A total of 8 hyperparameter combinations were evaluated using 5-fold cross-validation.

## 3. Best Hyperparameters

The best combination identified by GridSearchCV was:

- `n_estimators = 200`
- `learning_rate = 0.1`
- `max_depth = 3`

**Best Cross-Validation RMSE:** 27,964.82

## 4. Model Performance Comparison

The original and tuned Gradient Boosting models were evaluated on the same validation dataset.

| Metric | Original Model | Tuned Model   |
| ------ | -------------- | ------------- |
| MAE    | 16,360.67      | **15,864.42** |
| RMSE   | 26,772.24      | **26,239.98** |
| R²     | 0.9066         | **0.9102**    |

The tuned model achieved lower MAE and RMSE values and a higher R² score compared with the original model.

## 5. Model Saving

The optimized Gradient Boosting model was saved using Joblib.

**Saved model path:** `models/gradient_boosting_tuned.joblib`

The saved model was successfully reloaded and verified.

## 6. Conclusion

Hyperparameter tuning successfully improved the validation performance of the Gradient Boosting Regressor.

The tuned model achieved an RMSE of 26,239.98 and an R² score of approximately 0.9102.

Therefore, the tuned Gradient Boosting Regressor was selected as the final model for generating house-price predictions on the test dataset.
