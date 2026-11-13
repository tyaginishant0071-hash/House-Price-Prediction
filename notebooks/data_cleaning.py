import pandas as pd

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


# 2: Missing Value Analysis and Cleaning Strategy
# Calculating the number of missing values in each column of X_train:
print(f"Missing values in X_train: \n{X_train.isnull().sum()}\n")

# Filtering the columns that contain at least one missing value:
missing_cols= X_train.columns[X_train.isnull().any()]
print(f"Columns with at least one missing value in X_train: \n{missing_cols}\n")

# Counts and percentage of columns that contains missing value:
print(f"Missing value counts for X_train: \n{X_train.isnull().sum().sort_values(ascending= False)}\n")
print(f"Missing value percentage for X_train: \n{round((X_train.isnull().mean() * 100), 4).sort_values(ascending= False)}\n")