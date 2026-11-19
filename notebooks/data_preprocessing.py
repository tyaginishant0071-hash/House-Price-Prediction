import pandas as pd
import numpy as np
import joblib
import os

os.makedirs('models', exist_ok= True)

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


# 1: Load Feature-Engineered Datasets 
# Load and read the data from data/processed/ (X_train_fe.csv, X_val_fe.csv and X_test_fe.csv):
train_fe_df= pd.read_csv('data/processed/X_train_fe.csv')
val_fe_df= pd.read_csv('data/processed/X_val_fe.csv')
test_fe_df= pd.read_csv('data/processed/X_test_fe.csv')

# Printing the shape of each dataset:
print(f"Shape of train_fe_df: {train_fe_df.shape}\n")
print(f"Shape of val_fe_df: {val_fe_df.shape}\n")
print(f"Shape of test_fe_df: {test_fe_df.shape}\n")

# Checking for missing values if any:
print(f"Missing values in train_fe_df: {train_fe_df.isnull().sum().sum()}\n")
print(f"Missing values in val_fe_df: {val_fe_df.isnull().sum().sum()}\n")
print(f"Missing values in test_fe_df: {test_fe_df.isnull().sum().sum()}\n")


# 2: Identify Numerical and Categorical Features
# Identifying numerical columns in train_fe_df using select_dtypes():
train_fe_num_cols= train_fe_df.select_dtypes(include= np.number).columns

# Identifying categorical columns in train_fe_df using select_dtypes():
categorical= ['object', 'category']
train_fe_cat_cols= train_fe_df.select_dtypes(categorical).columns

# Printing numerical and categorical columns:
print(f"Numerical columns in train_fe_df: \n{len(train_fe_num_cols)}\n")
print(f"Categorical columns in train_fe_df: \n{len(train_fe_cat_cols)}\n")

# Total numerical columns and categorical columns = 84:
print(f"Total numerical columns + categorical columns: {len(train_fe_num_cols) + len(train_fe_cat_cols)}\n")


# 3: One-Hot Encoding
# Creating an encoder with handle_unknown= 'ignore':
one_hot_encoder= OneHotEncoder(handle_unknown= 'ignore', sparse_output= False)

# Setting sparse_output=False to obtain a dense array:
one_hot_encoder.set_output(transform= 'pandas')

# Fitting the encoder only the 43 categorical columns from "train_fe_df":
one_hot_encoder.fit(train_fe_df[train_fe_cat_cols])

# Transforming all the categorical columns of "train_fe_df", "val_fe_df" and "test_fe_df" using resuable function:
def transforming_categorical_columns(df):
  
  encoded_df= one_hot_encoder.transform(df[train_fe_cat_cols])

  return encoded_df

train_fe_encoded= transforming_categorical_columns(train_fe_df)
val_fe_encoded= transforming_categorical_columns(val_fe_df)
test_fe_encoded= transforming_categorical_columns(test_fe_df)

# Printing the shapes of "train_fe_encoded", "val_fe_encoded" and "test_fe_encoded":
print(f"Shape of train_fe_encoded: {train_fe_encoded.shape}\n")
print(f"Shape of val_fe_encoded: {val_fe_encoded.shape}\n")
print(f"Shape of test_fe_encoded: {test_fe_encoded.shape}\n")

# Verifying all the three encoded datasets have identical column names and order:
print(f"train_fe_encoded columns match: {train_fe_encoded.columns.equals(val_fe_encoded.columns)}\n")
print(f"train_fe_encoded columns match: {train_fe_encoded.columns.equals(test_fe_encoded.columns)}\n")
print(f"val_fe_encoded columns match: {val_fe_encoded.columns.equals(test_fe_encoded.columns)}\n")

# Checking missing values if any:
print(f"Missing values in train_fe_encoded: {train_fe_encoded.isnull().sum().sum()}\n")
print(f"Missing values in val_fe_encoded: {val_fe_encoded.isnull().sum().sum()}\n")
print(f"Missing values in test_fe_encoded: {test_fe_encoded.isnull().sum().sum()}\n")

# Verifying all the encoded columns have numerical datatype:
print(f"Data type of train_fe_encoded: {train_fe_encoded.dtypes}\n")
print(f"Data type of val_fe_encoded: {val_fe_encoded.dtypes}\n")
print(f"Data type of test_fe_encoded: {test_fe_encoded.dtypes}\n")

# Combining all the 41 cols with 265 encoded columns with the help of 'pd.concat':
train_fe_final= pd.concat([train_fe_df[train_fe_num_cols], train_fe_encoded], axis= 1)
val_fe_final= pd.concat([val_fe_df[train_fe_num_cols], val_fe_encoded], axis= 1)
test_fe_final= pd.concat([test_fe_df[train_fe_num_cols], test_fe_encoded], axis= 1)

# Shape of datasets:
print(f"Shape of train_fe_final: {train_fe_final.shape}\n")
print(f"Shape of val_fe_final: {val_fe_final.shape}\n")
print(f"Shape of test_fe_final: {test_fe_final.shape}\n")

# Verifying all the three datasets have identical columns:
print(f"train_fe_final and val_fe_final: {train_fe_final.columns.equals(val_fe_final.columns)}\n")
print(f"train_fe_final and test_fe_final: {train_fe_final.columns.equals(test_fe_final.columns)}\n")
print(f"val_fe_final and test_fe_final: {val_fe_final.columns.equals(test_fe_final.columns)}\n")

# Checking missing values in all three datasets:
print(f"Missing values in train_fe_final: {train_fe_final.isnull().sum().sum()}\n")
print(f"Missing values in val_fe_final: {val_fe_final.isnull().sum().sum()}\n")
print(f"Missing values in test_fe_final: {test_fe_final.isnull().sum().sum()}\n")

# Verifying all the 306 columns are numerical in the datasets:
print(f"Train_fe_final all numeric: {train_fe_final.select_dtypes(include= [np.number]).shape[1] == train_fe_final.shape[1]}\n")
print(f"Val_fe_final all numeric: {val_fe_final.select_dtypes(include= [np.number]).shape[1] == val_fe_final.shape[1]}\n")
print(f"Test_fe_final all numeric: {test_fe_final.select_dtypes(include= [np.number]).shape[1] == test_fe_final.shape[1]}\n")

# Shape of all three datasets:
print(f"Shape of train_fe_final: {train_fe_final.shape}\n")
print(f"Shape of val_fe_final: {val_fe_final.shape}\n")
print(f"Shape of test_fe_final: {test_fe_final.shape}\n")


# 4: Numerical Feature Scaling
# Creating a "StandardScaler" object:
st_scaler= StandardScaler()

# Fitting the scaler on 41 original numerical columns:
st_scaler.fit(train_fe_df[train_fe_num_cols]) 

# Creating a reusable function to transform numerical columns:
def transforming_numerical_columns(num_cols):
  
  scaled_num_cols= st_scaler.transform(num_cols)     # Using "st_scaler.transform" for transforming the numerical columns.

  scaled_num_cols_df= pd.DataFrame(scaled_num_cols, columns= train_fe_num_cols, index= num_cols.index)    # Converting scaled arrays into pandas dataframe.

  return scaled_num_cols_df

# Applying function to (training, validation and test) datasets:
scaled_train_num_cols= transforming_numerical_columns(train_fe_df[train_fe_num_cols])
scaled_val_num_cols= transforming_numerical_columns(val_fe_df[train_fe_num_cols])
scaled_test_num_cols= transforming_numerical_columns(test_fe_df[train_fe_num_cols])

# Printing the shape of all three scaled datasets:
print(f"Shape of scaled_train_num_cols: {scaled_train_num_cols.shape}\n")
print(f"Shape of scaled_val_num_cols: {scaled_val_num_cols.shape}\n")
print(f"Shape of scaled_test_num_cols: {scaled_test_num_cols.shape}\n")

# Verifying all columns are numerical:
print(f"Columns of 'scaled_train_num_cols' are numerical: {scaled_train_num_cols.select_dtypes(include= np.number).shape[1] == scaled_train_num_cols.shape[1]}\n")
print(f"Columns of 'scaled_val_num_cols' are numerical: {scaled_val_num_cols.select_dtypes(include= np.number).shape[1] == scaled_val_num_cols.shape[1]}\n")
print(f"Columns of 'scaled_test_num_cols' are numerical: {scaled_test_num_cols.select_dtypes(include= np.number).shape[1] == scaled_test_num_cols.shape[1]}\n")

# Missing value if any:
print(f"Missing values in 'scaled_train_num_cols': {scaled_train_num_cols.isnull().sum().sum()}\n")
print(f"Missing values in 'scaled_val_num_cols': {scaled_val_num_cols.isnull().sum().sum()}\n")
print(f"Missing values in 'scaled_test_num_cols': {scaled_test_num_cols.isnull().sum().sum()}\n")

# Verifying scaled DataFrame indexes match their original dataset:
print(f"'scaled_train_num_cols' dataframe indexes is equal to 'train_fe_df' original dataframe indexes: {scaled_train_num_cols.index.equals(train_fe_df.index)}\n")
print(f"'scaled_val_num_cols' dataframe indexes is equal to 'val_fe_df' original dataframe indexes: {scaled_val_num_cols.index.equals(val_fe_df.index)}\n")
print(f"'scaled_test_num_cols' dataframe indexes is equal to 'test_fe_df' original dataframe indexes: {scaled_test_num_cols.index.equals(test_fe_df.index)}\n")

# Combining "scaled_train_num_cols" and "train_fe_encoded" using (pd.concat(axis= 1)):
train_preprocessed= pd.concat([scaled_train_num_cols, train_fe_encoded], axis= 1)
val_preprocessed= pd.concat([scaled_val_num_cols, val_fe_encoded], axis= 1)
test_preprocessed= pd.concat([scaled_test_num_cols, test_fe_encoded], axis= 1)

# Printing the shape:
print(f"Shape of train_preprocessed: {train_preprocessed.shape}\n")
print(f"Shape of val_preprocessed: {val_preprocessed.shape}\n")
print(f"Shape of test_preprocessed: {test_preprocessed.shape}\n")


# 5: Verify Final Preprocessed Datasets
# Verifying all three datasets have identical columns names and order:
print(f"train_preprocessed and val_preprocessed: {train_preprocessed.columns.equals(val_preprocessed.columns)}\n")
print(f"train_preprocessed and test_preprocessed: {train_preprocessed.columns.equals(test_preprocessed.columns)}\n")
print(f"val_preprocessed and test_preprocessed: {val_preprocessed.columns.equals(test_preprocessed.columns)}\n")

# Checking all three datasets containing zero missing value:
print(f"Missing values in train_preprocessed: {train_preprocessed.isnull().sum().sum()}\n")
print(f"Missing values in val_preprocessed: {val_preprocessed.isnull().sum().sum()}\n")
print(f"Missing values in test_preprocessed: {test_preprocessed.isnull().sum().sum()}\n")

# Verifying all three datasets having 306 columns are numerical:
print(f"train_preprocessed have numerical columns: {train_preprocessed.select_dtypes(include= np.number).shape[1] == train_preprocessed.shape[1]}\n")
print(f"val_preprocessed have numerical columns: {val_preprocessed.select_dtypes(include= np.number).shape[1] == val_preprocessed.shape[1]}\n")
print(f"test_preprocessed have numerical columns: {test_preprocessed.select_dtypes(include= np.number).shape[1] == test_preprocessed.shape[1]}\n")

# Shape of all three datasets:
print(f"Shape of train_preprocessed: {train_preprocessed.shape}\n")
print(f"Shape of val_preprocessed: {val_preprocessed.shape}\n")
print(f"Shape of test_preprocessed: {test_preprocessed.shape}\n")

# Checking no duplicate column exist:
print(f"Duplicate columns in train_preprocessed: {train_preprocessed.columns.duplicated().sum()}\n")
print(f"Duplicate columns in val_preprocessed: {val_preprocessed.columns.duplicated().sum()}\n")
print(f"Duplicate columns in test_preprocessed: {test_preprocessed.columns.duplicated().sum()}\n")


# Saving 'train_preprocessed' as "X_train_preprocessed.csv":
train_preprocessed.to_csv('data/processed/X_train_preprocessed.csv', index= False)

# Saving 'val_preprocessed' as "X_val_preprocessed.csv":
val_preprocessed.to_csv('data/processed/X_val_preprocessed.csv', index= False)

# Saving 'test_preprocessed' as "X_test_preprocessed.csv":
test_preprocessed.to_csv('data/processed/X_test_preprocessed.csv', index= False)

# Saving the fitted "one_hot_encoder" using joblib:
joblib.dump(one_hot_encoder, 'models/one_hot_encoder.joblib')

# Saving the fitted "st_scaler" using joblib:
joblib.dump(st_scaler, 'models/standard_scaler.joblib')

# Saving "train_fe_num_cols" using joblib:
joblib.dump(train_fe_num_cols, 'models/numerical_columns.joblib')

# Saving "train_fe_cat_cols" using joblib:
joblib.dump(train_fe_cat_cols, 'models/categorical_columns.joblib')