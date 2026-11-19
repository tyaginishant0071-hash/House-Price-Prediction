import pandas as pd 
import numpy as np
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


# 1: Load Model Training Data
# Loading and Reading data from "data/processed":
X_train_pre_df= pd.read_csv('data/processed/X_train_preprocessed.csv')
X_val_pre_df= pd.read_csv('data/processed/X_val_preprocessed.csv')
y_train_df= pd.read_csv('data/processed/y_train.csv')
y_val_df= pd.read_csv('data/processed/y_val.csv')

# Printing the shape of all four datasets:
print(f"Shape of X_train_pre_df: {X_train_pre_df.shape}\n")
print(f"Shape of X_val_pre_df: {X_val_pre_df.shape}\n")
print(f"Shape of y_train_df: {y_train_df.shape}\n")
print(f"Shape of y_val_df: {y_val_df.shape}\n")

# 2: Train Your First Linear Regression Model
# Creating a "LinearRegression" object model:
LR_model= LinearRegression()

# Fitting the "X_train_pre_df" and "y_train_df":
LR_model.fit(X_train_pre_df, y_train_df)

# Generating predictions using "X_val_pre_df":
y_val_pred= LR_model.predict(X_val_pre_df)

# Printing the first 5 predicted house prices:
print(f"First 5 predicted house prices: \n{y_val_pred[:5]}\n")

# Model Evaluation
# Calculating MAE using (y_val_df and y_val_pred):
lr_mae= mean_absolute_error(y_val_df, y_val_pred)
print(f"Mean Absolute Error: {lr_mae}\n")

# Calculating MSE using (y_val_df and y_val_pred):
lr_mse= mean_squared_error(y_val_df, y_val_pred)
print(f"Mean Squared Error: {lr_mse}\n")

# Calculating RMSE using (np.sqrt(mse)):
lr_rmse= np.sqrt(lr_mse)
print(f"Root Mean Squared Error: {lr_rmse}\n")

# Calculating r2_score using (y_val_df and y_val_pred):
lr_r2_s= r2_score(y_val_df, y_val_pred)
print(f"R2 score: {lr_r2_s}\n")


# 3: Random Forest Regressor
# Creating a "RandomForestRegressor" object with 'n_estimators= 100 and random_state= 42':
RF_model= RandomForestRegressor(n_estimators= 100, random_state= 42)

# Fitting the model using (X_train_pre_df and y_train_df):
RF_model.fit(X_train_pre_df, y_train_df.squeeze())

# Generating predictions using "X_val_pre_df":
RF_y_val_pred= RF_model.predict(X_val_pre_df)

# Printing first 5 predicted house price:
print(f"First 5 predicted house prices: \n{RF_y_val_pred[:5]}\n")

# Evaluate Random Forest
# Calculating MAE using (y_val_df and RF_y_val_pred):
rf_mae= mean_absolute_error(y_val_df, RF_y_val_pred)
print(f"Mean Absolute Error: {rf_mae}\n")

# Calculating MSE using (y_val_df and RF_y_val_pred):
rf_mse= mean_squared_error(y_val_df, RF_y_val_pred)
print(f"Mean Squared Error: {rf_mse}\n")

# Calculating RMSE using (np.sqrt(mse)):
rf_rmse= np.sqrt(rf_mse)
print(f"Root Mean Squared Error: {rf_rmse}\n")

# Calculating r2_score using (y_val_df and RF_y_val_pred):
rf_r2_s= r2_score(y_val_df, RF_y_val_pred)
print(f"R2 score: {rf_r2_s}\n")


# 4: Train Gradient Boosting Regressor
# Creating a "GradientBoostingRegressor" object with 'n_estimators= 100 and random_state= 42':
GB_model= GradientBoostingRegressor(n_estimators= 100, random_state=42)

# Fitting the model using (X_train_pre_df and y_train_df):
GB_model.fit(X_train_pre_df, y_train_df.squeeze())

# Generating predictions using "X_val_pre_df":
GB_y_val_pred= GB_model.predict(X_val_pre_df)

# Printing first 5 predicted house prices:
print(f"First 5 predicted house prices: \n{GB_y_val_pred[:5]}\n")

# Evaluate Gradient Boosting
# Calculating MAE using (y_val_df and GB_y_val_pred):
gb_mae= mean_absolute_error(y_val_df, GB_y_val_pred)
print(f"Mean Absolute Error: {gb_mae}\n")

# Calculating MSE using (y_val_df and GB_y_val_pred):
gb_mse= mean_squared_error(y_val_df, GB_y_val_pred)
print(f"Mean Squared Error: {gb_mse}\n")

# Calculating RMSE using (np.sqrt(mse)):
gb_rmse= np.sqrt(gb_mse)
print(f"Root Mean Squared Error: {gb_rmse}\n")

# Calculating r2_score using (y_val_df and GB_y_val_pred):
gb_r2_s= r2_score(y_val_df, GB_y_val_pred)
print(f"R2 score: {gb_r2_s}\n")


# 5: Create Model Comparison Table
# Create Model Comparison Table
# Creating a pandas DataFrame named "model_comparison":
model_comparison= pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'Gradient Boosting'],
    'MAE': [lr_mae, rf_mae, gb_mae],
    'RMSE': [lr_rmse, rf_rmse, gb_rmse],
    'R2_score': [lr_r2_s, rf_r2_s, gb_r2_s]
}).sort_values(by= 'RMSE', ascending= True)

# Printing the "model_comparison" DataFrame:
print(f"Model Comparison Table: \n{model_comparison}\n")


# 6: Hyperparameter Tuning
# Creating a parameter named "param_grid":
param_grid= {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [2, 3]
}

# Printing the "param_grid":
print(f"Parameter Grid: \n{param_grid}\n")

# Creating a "GridSearchCV" object:
grid_search= GridSearchCV(estimator= GB_model, param_grid= param_grid, cv= 5, scoring= 'neg_root_mean_squared_error', n_jobs= -1, refit= True)

# Fitting "grid_search" using (X_train_pre_df and y_train_df.squeeze()):
grid_search.fit(X_train_pre_df, y_train_df.squeeze())

# Printing "grid_search" 'best_param' to identify the best hyperparameters:
best_param= grid_search.best_params_
print(f"Best Hyperparameters: \n{best_param}\n")

# Printing "grid_search" 'best_score' to inspect the best cross-validation score:
best_score= grid_search.best_score_
print(f"Best Cross-Validation Score: {best_score}\n")

# Evaluate the Tuned Model
# Retrieving the tuned model using (grid_search.best_estimator_):
tuned_model= grid_search.best_estimator_

# Generating predictions using "X_val_pre_df":
tuned_y_val_pred= tuned_model.predict(X_val_pre_df)

# Calculating MAE:
tuned_mae= mean_absolute_error(y_val_df, tuned_y_val_pred)
print(f"Mean Absolute Error: {tuned_mae}\n")

# Calculating RMSE:
tuned_rmse= np.sqrt(mean_squared_error(y_val_df, tuned_y_val_pred))
print(f"Root Mean Squared Error: {tuned_rmse}\n")

# Calculating R2 score:
tuned_r2_s= r2_score(y_val_df, tuned_y_val_pred)
print(f"R2 score: {tuned_r2_s}\n")

# Saving the "tuned model":
joblib.dump(tuned_model, 'models/gradient_boosting_tuned.joblib')

# Loading the saved model:
loaded_model= joblib.load('models/gradient_boosting_tuned.joblib')

# Printing the loaded_model:
print(f"Loaded model: {loaded_model}")
