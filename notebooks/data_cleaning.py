import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split


# 1: Dataset Preparation and Train–Validation Split
# Loading and reading the "train" and "test" csv files:
train_df= pd.read_csv("data/raw/train.csv")
test_df= pd.read_csv("data/raw/test.csv")

# Separating "SalePrice" as the target variable (y) from "train_df":
y= train_df['SalePrice']

# Creating (X) by excluding "Id" and "SalePrice":
X= train_df.drop(columns= ['Id', 'SalePrice'])

# Spliting X and y into 80% training and 20% validation data using (random_state= 42):
X_train, X_val, y_train, y_val= train_test_split(X, y, test_size= 0.2, random_state= 42)

# Checking the shape of the training and validation:
print(f"Shape of X_train: {X_train.shape}\n")
print(f"Shape of X_val: {X_val.shape}\n")
print(f"Shape of y_train: {y_train.shape}\n")
print(f"Shape of y_val: {y_val.shape}\n")


# 2: Handle categorical missing values
# Creating a copy of X_train:
X_train_copy= X_train.copy()

# Identifying categorical columns:
categorical= ['object', 'category']
categorical_columns= X_train_copy.select_dtypes(categorical).columns
print(f"Categorical columns: \n{categorical_columns}\n")

# Missing categorical columns:
missing_cat_cols= X_train_copy[categorical_columns].columns[X_train_copy[categorical_columns].isnull().any()]
print(f"Missing categorical columns: \n{missing_cat_cols}\n")

# Filling all missing columns with "Unknown":
X_train_copy[missing_cat_cols]= X_train_copy[missing_cat_cols].fillna("Unknown")

print("Missing value after handled:", X_train_copy[missing_cat_cols].isnull().sum().sum())


# 3: Handle Numerical Missing Values
# Identifying numerical columns:
numerical_columns= X_train_copy.select_dtypes(include= np.number).columns

# Identifying numerical columns containing missing value:
missing_numerical_cols= X_train_copy[numerical_columns].columns[X_train_copy[numerical_columns].isnull().any()]
print(f"Missing numerical columns: \n{missing_numerical_cols}\n")

# Calculating and filling the missing values with the median:
train_medians = X_train[numerical_columns].median()
X_train_copy[numerical_columns] = X_train_copy[numerical_columns].fillna(train_medians)

print(f"Missing value after handled: {X_train_copy[missing_numerical_cols].isnull().sum().sum()}\n")


# 4: Verify Cleaned Training Data
# Checking total remaining missing value in "X_train_copy":
print(f"Total remaining missing value in X_train_copy: {X_train_copy.isnull().sum().sum()}\n")

# Printing the shape of "X_train_copy":
print(f"Shape of X_train_copy: {X_train_copy.shape}\n")

# Checking original "X_train" remain unchanged:
print(f"Shape of X_train: {X_train.shape}\n")
print(f"Missing value in X_train: {X_train.isnull().sum().sum()}\n")


# 5: Clean Validation Data
# Creating a copy of "X_val":
X_val_copy= X_val.copy()

# Missing values in "X_val":
print(f"Missing value in X_val: {X_val.isnull().sum().sum()}\n")

# Filling categorical columns missing value with "Unknown":
X_val_copy[missing_cat_cols]= X_val_copy[missing_cat_cols].fillna("Unknown")

# Filling numerical columns missing value with "missing_numerical_cols":
X_val_copy[numerical_columns] = X_val_copy[numerical_columns].fillna(train_medians)

# Verifying missing values in "X_val_copy":
print(f"Missing value in X_val_copy: {X_val_copy.isnull().sum().sum()}\n")

# Verifying shape:
print(f"Shape of X_val_copy: {X_val_copy.shape}\n")


# 6: Clean Kaggle Test Data
# Creating a copy of "test_df":
test_df_copy= test_df.copy()

# Removing "Id" from "test_df_copy" features:
test_df_copy= test_df_copy.drop(columns= ['Id'])

# Preserved the (test_df_copy= Id)
test_ids = test_df['Id'].copy()

# Checking missing columns in "test_df_copy":
missing_test_cols= test_df_copy.columns[test_df_copy.isnull().any()]
print(f"Missing columns in test_df_copy: \n{missing_test_cols}\n")
print(f"Total missing columns in test_df_copy: {len(missing_test_cols)}\n")

# Total missing values:
print(f"Total missing values in test_df_copy: {test_df_copy.isnull().sum().sum()}\n")

# Separating categorical colums in "missing test cols":
categorical_test_cols= test_df_copy[missing_test_cols].select_dtypes(categorical).columns
print(f"Categorical columns in missing test cols: \n{categorical_test_cols}\n")
print(f"Total categorical columns in missing test columns: {len(categorical_test_cols)}\n")

# Separating numerical columns in "missing test cols":
numerical_test_cols= test_df_copy[missing_test_cols].select_dtypes(include= np.number).columns
print(f"Numerical columns in missing test cols: \n{numerical_test_cols}\n")
print(f"Total numerical columns in missing test columns: {len(numerical_test_cols)}\n")

# Filling missing categorical columns with "Unknown":
test_df_copy[categorical_test_cols]= test_df_copy[categorical_test_cols].fillna("Unknown")

# Filling missing numerica colums with "median" of "missing numerical cols":
test_df_copy[numerical_test_cols]= test_df_copy[numerical_test_cols].fillna(train_medians)

# After handling missing values in both categorical and numerical columns:
print(f"Total missing values in test_df_copy after handling: {test_df_copy.isnull().sum().sum()}\n")

# Verifying the final test feature shape:
print(f"Shape of test_df_copy: {test_df_copy.shape}\n")


# 7: Save Cleaned Datasets in "processed"
# Save X_train_copy as "X_train_clean.csv":
X_train_copy.to_csv('data/processed/X_train_clean.csv', index= False)

# Save X_val_copy as "X_val_clean.csv":
X_val_copy.to_csv('data/processed/X_val_clean.csv', index= False)

# Save test_df_copy as "X_test_clean.csv":
test_df_copy.to_csv('data/processed/X_test_clean.csv', index= False)

# Save "y_train" as "y_train.csv": 
y_train.to_csv('data/processed/y_train.csv', index=False)

# Save "y_val" as "y_val.csv":
y_val.to_csv('data/processed/y_val.csv', index=False)

# Save "test_ids" as "test_ids.csv"
test_ids.to_csv('data/processed/test_ids.csv', index=False)