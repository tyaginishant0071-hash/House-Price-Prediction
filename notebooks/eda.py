import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1: Load and inspect the dataset
# Dataset loaded:
train_df= pd.read_csv('data/raw/train.csv')
test_df= pd.read_csv('data/raw/test.csv')

# Printing the first 5 rows of the "train dataset":
print(f"First 5 rows of the train dataset: \n{train_df.head()}\n")

# Checking the shape of both the datasets:
print(f"Shape of the train dataset: {train_df.shape}\n")
print(f"Shape of the test dataset: {test_df.shape}\n")

# 2: Data Types and Basic Statistics
# Checking the columns name in the training dataset:
print(f"Columns name in the train dataset: \n{train_df.columns}\n")

# Checking the data types of all columns:
print(f"Data types of all columns: \n{train_df.dtypes}\n")

# Identifying numerical columns:
print(f"Numerical columns: \n{train_df.select_dtypes(include= np.number)}\n")

# Identifying categorical columns:
categorical = ['object', 'category']
print(f"Categorical columns: \n{train_df.select_dtypes(include= categorical)}\n")

# Generating descriptive statistics for numerical columns:
numerical_cols= train_df.select_dtypes(include= np.number)
print(f"Descriptive statistics for numerical columns: \n{numerical_cols.describe()}\n")

# Generating descriptive statistics for categorical features:
categorical_cols= train_df.select_dtypes(include= categorical)
print(f"Descriptive statistics for categorical columns: \n{categorical_cols.describe()}\n")

# 3: Missing Value Analysis
# Calculating missing values for each training column:
print(f"Missing values for each training column: \n{train_df.isnull().sum()}\n")

# Calculating missing-value percentages:
print(f"Missing values percentage for each training column: \n{(train_df.isnull().mean() * 100).round(2)}\n")

# Displaying columns with at least one missing value:
print(f"Columns with at least one missing value: \n{train_df.columns[train_df.isnull().any()]}\n")

# Sorting missing value counts in descending order:
missing= train_df.isnull().sum()
sort_missing= missing.sort_values(ascending= False)
print(f"Missing value counts in descending order: \n{sort_missing}\n")

# Identifying the missing values more than 40%:
missing_val_above_40_per= train_df.columns[train_df.isnull().mean() > 0.40]
print(f"Columns with missing values more than 40%: \n{missing_val_above_40_per}\n")

# Checking the missing values in the "test" dataset:
print(f"Missing values in the test dataset: \n{test_df.isnull().sum()}\n")

# Checking the "SalePrice" has missing value:
print(f"Missing value in SalePrice: \n{train_df['SalePrice'].isnull().sum()}\n")

# 4: Numerical Feature Analysis
# Count of numerical columns in "train_df":
print(f"Count of numerical columns in train_df: {len(numerical_cols.columns)}\n")

# Analyzing the distribution of "SalePrice" by using 'histogram':
plt.figure(figsize= (10, 8))
sns.histplot(train_df['SalePrice'], bins= 30, kde= True)
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.ylabel('Frequency')
plt.show()

# Creating a 'boxplot' of the "SalePrice" to identify the potential outlier:
plt.figure(figsize= (10, 8))
sns.boxplot(train_df['SalePrice'])
plt.title('Boxplot of SalePrice')
plt.xlabel('SalePrice')
plt.show()

# Calculating the skewness of "SalePrice":
print(f"Skewness of SalePrice: {train_df['SalePrice'].skew():.4f}\n")

# Creating a 'histograms' for "LotArea", "GrLivArea", "TotalBsmtSF", and "GarageArea":
cols_to_plot= ['LotArea', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']

fig, axes= plt.subplots(nrows= 2, ncols= 2, figsize= (14, 10))
axes= axes.flatten()

for col, ax in zip(cols_to_plot, axes):
  sns.histplot(train_df[col], bins= 30, kde= True, ax= ax)
  ax.set_title(f'Distribution of {col}')
  ax.set_xlabel(col)
  ax.set_ylabel('Frequency')

plt.tight_layout()
plt.show()

# Creating 'boxplots' for "LotArea", "GrLivArea", "TotalBsmtSF", and "GarageArea":
cols_to_boxplot= ['LotArea', 'GrLivArea', 'TotalBsmtSF', 'GarageArea']

figure, boxplot_axes= plt.subplots(nrows= 2, ncols= 2, figsize= (14, 10))
boxplot_axes= boxplot_axes.flatten()

for cols, axs in zip(cols_to_boxplot, boxplot_axes):
  sns.boxplot(y=train_df[cols], ax= axs)
  axs.set_title(f'Boxplot of {cols}')
  axs.set_ylabel(cols)

plt.tight_layout()
plt.show()

# Calculating correlations between "numerical features" and "SalePrice":
corr_matrix= train_df.select_dtypes(include= np.number).corr()
corr_with_saleprice= corr_matrix['SalePrice'].sort_values(ascending= False)
print(f"Correlation with SalePrice: \n{corr_with_saleprice}\n")

#Displaying the 10 features with the strongest positive correlations with "SalePrice":
top_10_pos_corr= corr_with_saleprice.iloc[1:11]
print(f"Top 10 features with the strongest positive correlations with SalePrice: \n{top_10_pos_corr}\n")

#Displaying the 10 features with the strongest negative correlations with "SalePrice":
top_10_neg_corr= corr_with_saleprice.iloc[::-1].head(10)
print(f"Top 10 features with the strongest negative correlations with SalePrice: \n{top_10_neg_corr}\n")