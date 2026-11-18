import pandas as pd 
import numpy as np

# 1: Load and Inspect Cleaned Data
# Loading and reading the file from "data/processed":
clean_test_df= pd.read_csv("data/processed/X_test_clean.csv")
clean_train_df= pd.read_csv("data/processed/X_train_clean.csv")
clean_val_df= pd.read_csv("data/processed/X_val_clean.csv")

# Printing the shape of all 3 datasets:
print(f"Shape of clean_test_df: {clean_test_df.shape}\n")
print(f"Shape of clean_train_df: {clean_train_df.shape}\n")
print(f"Shape of clean_val_df: {clean_val_df.shape}\n")

# Checking missing values:
print(f"Missing values in clean_test_df: {clean_test_df.isnull().sum().sum()}\n")
print(f"Missing values in clean_train_df: {clean_train_df.isnull().sum().sum()}\n")
print(f"Missing values in clean_val_df: {clean_val_df.isnull().sum().sum()}\n")


# 2: Identify Numerical and Categorical Features
# Identifying numerical columns in "X_train_clean" using select_dtypes():
X_train_clean_num_cols= clean_train_df.select_dtypes(include= np.number).columns

# Identifying categorical columns in X_train_clean using select_dtypes():
categorical= ['object', 'category']
X_train_clean_cat_cols= clean_train_df.select_dtypes(categorical).columns

# Counting the total numerical columns and categorical columns:
print(f"Total numerical columns in X_train_clean: {len(X_train_clean_num_cols)}\n")
print(f"Total categorical columns in X_train_clean: {len(X_train_clean_cat_cols)}\n")

# Verifying the numerical columns + categorica columns =  79:
print(f"Total columns numerical + categorical: {len(X_train_clean_num_cols) + len(X_train_clean_cat_cols)} \n")

# Printing the columns name and count:
print(f"Numerical columns in X_train_clean: \n{X_train_clean_num_cols}: {len(X_train_clean_num_cols)}\n")
print(f"Categorical columns in X_train_clean: \n{X_train_clean_cat_cols}: {len(X_train_clean_cat_cols)}\n")


# 3: Create New Features
# Creating a function "feature_engineering(df)":
def feature_engineering(df):
  
  # Creating a copy of the dataset(df):
  df_copy= df.copy()

  # Creating new columns:
  df_copy['TotalSF']= df_copy['TotalBsmtSF'] + df_copy['GrLivArea']
  df_copy['TotalBath']= df_copy['FullBath'] + (df_copy['HalfBath'] * 0.5) + df_copy['BsmtFullBath'] + (df_copy['BsmtHalfBath'] * 0.5)
  df_copy['HouseAge']= df_copy['YrSold'] - df_copy['YearBuilt']
  df_copy['RemodAge']= df_copy['YrSold'] - df_copy['YearRemodAdd']
  df_copy['TotalPorchSF']= df_copy['OpenPorchSF'] + df_copy['EnclosedPorch'] + df_copy['3SsnPorch'] + df_copy['ScreenPorch']

  # Removing negative value from "HouseAge" and "RemodAge":
  df_copy.loc[df_copy['HouseAge'] < 0, 'HouseAge']= 0
  df_copy.loc[df_copy['RemodAge'] < 0, 'RemodAge']= 0

  return df_copy

train_fe = feature_engineering(clean_train_df)
val_fe = feature_engineering(clean_val_df)
test_fe = feature_engineering(clean_test_df)

# Checking total columns and missing values if any:
print(f"Shape of clean_test_df_copy: {train_fe.shape}  |  Missing values: {train_fe.isnull().sum().sum()}\n")
print(f"Shape of clean_val_df_copy: {val_fe.shape}  |  Missing value: {val_fe.isnull().sum().sum()}\n")
print(f"Shape of clean_test_df_copy: {test_fe.shape}  | Missing value: {test_fe.isnull().sum().sum()}\n")


# 4: Final Verification
# Verify identical columns across all three datasets:
print(f"Train and Val columns match: {train_fe.columns.equals(val_fe.columns)}\n")
print(f"Train and Test columns match: {train_fe.columns.equals(test_fe.columns)}\n")
print(f"Val and Test columns match: {val_fe.columns.equals(test_fe.columns)}\n")

# Checking negative values in "HouseAge" and "RemodAge":
for name, df in [
    ('Train', train_fe),
    ('Val', val_fe),
    ('Test', test_fe)]:

    print(name, "Negative HouseAge", (df['HouseAge'] < 0).sum())
    print(name, "Negative RemodAge", (df['RemodAge'] < 0).sum())


# 5: Inspect Negative Age Records
# Displaying rows in "train_fe" where 'RemodeAge' is negative:
train_RemodeAge_neg_row= train_fe[train_fe['RemodAge'] < 0]

# Displaying rows in "test_fe" where 'HouseAge' and 'RemodeAge' is negative:
test_RemodeAge_neg_row= test_fe[(test_fe['RemodAge'] < 0) | (test_fe['HouseAge'] < 0)]

# Shows only ('YrSold', 'YearBuilt', 'YearRemodAdd', 'HouseAge' and 'RemodAge')  for those rows:
print(f"'train_fe' Rows where 'RemodAge' is negative: \n{train_RemodeAge_neg_row[['YrSold', 'YearBuilt', 'YearRemodAdd', 'HouseAge', 'RemodAge']]}\n")
print(f"'test_fe' Rows where 'HouseAge' and 'RemodAge' is negative: \n{test_RemodeAge_neg_row[['YrSold', 'YearBuilt', 'YearRemodAdd', 'HouseAge', 'RemodAge']]}\n")


# 6: Handle Negative Age Values
# Negative values in both Age are handled in functions
# Verifying that both Age columns contain zero negative values:
print(f"Negative value in 'HouseAge': {(train_fe['HouseAge'] < 0).sum()}\n")
print(f"Negative value in 'RemodAge': {(train_fe['RemodAge'] < 0).sum()}\n")

# Confirming all the datasets have 84 columns:
print(f"Shape of train_fe: {train_fe.shape}\n")
print(f"Shape of val_fe: {val_fe.shape}\n")
print(f"Shape of test_fe: {test_fe.shape}\n")

# Checking negative values in "HouseAge" and "RemodAge" in "val_fe" and "test_fe":
print(f"Negative value in 'HouseAge' in 'val_fe': {(val_fe['HouseAge'] < 0).sum()}\n")
print(f"Negative value in 'RemodAge' in 'val_fe': {(val_fe['RemodAge'] < 0).sum()}\n")
print(f"Negative value in 'HouseAge' in 'test_fe': {(test_fe['HouseAge'] < 0).sum()}\n")
print(f"Negative value in 'RemodAge' in 'test_fe': {(test_fe['RemodAge'] < 0).sum()}\n")


# 7: Save Engineered Datasets
# Saving 'train_fe' as "X_train_fe.csv":
train_fe.to_csv('data/processed/X_train_fe.csv', index= False)

# Saving 'val_fe' as "X_val_fe.csv":
val_fe.to_csv('data/processed/X_val_fe.csv', index= False)

# Saving 'test_fe' as "X_test_fe.csv":
test_fe.to_csv('data/processed/X_test_fe.csv', index= False)