import pandas as pd 
import numpy as np
import joblib


# 1: Load Test Dataset and Saved Model
# Loading and Reading the test dataset:
test_data_df= pd.read_csv('data/processed/X_test_preprocessed.csv')

# Loading the saved model:
GBT_model= joblib.load('models/gradient_boosting_tuned.joblib')

# Printing the shape of the test dataset:
print(f"Shape of the test dataset: {test_data_df.shape}\n")

# Printing the loaded model details:
print(f"Loaded Gradient Boosting Model: {GBT_model}\n")


# 2: Generate Final House Price Predictions
# Calling the model's predict() method using the preprocessed test dataset:
test_predictions= GBT_model.predict(test_data_df)

# Printing the shape of the test predictions:
print(f"Shape of the test predictions: {test_predictions.shape}\n")

# Printing the first 5 predictions:
print(f"First 5 House Price Predictions: \n{test_predictions[:5]}\n")

# 3: Prediction Validation
# Printing "GBT_model.n_estimators" and confirming it equals 200:
print(f"GBT_model(n_estimators): {GBT_model.n_estimators}\n")

# Verifying the "test_data_df" shape:
print(f"Shape of the test dataset: {test_data_df.shape}\n")

# Checking "test_predictions" for any NaN values:
print(f"NaN values in test_predictions: {np.isnan(test_predictions).any()}\n")

# Checking "test_predictions" having any infinite values:
print(f"Infinite values in test_predictions: {np.isinf(test_predictions).any()}\n")

# Checking whether any "test_predictions" have house prices as zero or negative:
print(f"Test predictions having zero or negative values: {(test_predictions <= 0).any()}\n") 

# Confirming that the test dataset columns match the model's training feature names and order using 'feature_names_in_':
print(f"Test dataset columns match the model's training feature names and order: {np.array_equal(test_data_df.columns, GBT_model.feature_names_in_)}\n")


# 4: Create Kaggle Submission File
# Loading the "original test IDs" from "data/processed":
# Loading the "original test dataframe" from "data/raw":
test_ids= pd.read_csv('data/processed/test_ids.csv')
test_df= pd.read_csv('data/raw/test.csv')

# Printing the shape of "test_ids":
print(f"Total test IDs: {test_ids.shape[0]}\n")

# Verifying that the IDs containing 1459 rows and original test dataset order:
print(f"Exactly 1459 rows: {test_ids.shape[0] == 1459}\n")

# Verifying "test_ids" equal to "test_df[Id]":
print(f"test_ids equals to test_df[Id]: {test_ids['Id'].equals(test_df['Id'])}\n")

# Creating a dataframe named as "submission_df":
submission_df= pd.DataFrame({
    'Id': test_ids['Id'],
    'SalePrice': np.round(test_predictions, 4)
})

# Printing first 5 rows of the dataframe:
print(f"First 5 rows of the dataframe: \n{submission_df.head()}\n")

# Printing the shape of "submission_df":
print(f"Shape of the submission_df: {submission_df.shape}\n")

# Columns of "submission_df":
print(f"Columns of the submission_df: {submission_df.columns}\n")

# Saving the "submission_df" as "submission.csv":
submission_df.to_csv('data/processed/submission.csv', index= False)

# Missing values in "submission_df":
print(f"Missing values in submission_df: {submission_df.isnull().sum().sum()}\n")