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


# 5: Categorical Feature Analysis
# Counting the total categorical columns:
print(f"Total categorical columns: {len(categorical_cols.columns)}\n")

# Displaying the count of unique value in each categorical column:
print(f"Unique values in each categorical column: \n{categorical_cols.nunique(dropna= False)}\n")

# Identifying the categorical column with 'one' unique value:
unique_counts = categorical_cols.nunique()
one_unique_value = unique_counts[unique_counts == 1]
print(f"Categorical column with 'one' unique value: \n{len(one_unique_value)}\n")

# Displaying value counts for "MSZoning", "Neighborhood", "HouseStyle", and "SaleCondition":
value_counts_cols= ['MSZoning', 'Neighborhood', 'HouseStyle', 'SaleCondition']
for cols in value_counts_cols:
  value_count= categorical_cols[cols].value_counts()
  print(f"Value counts for the selected columns: \n{ value_count}\n")

# Creating countplots for "MSZoning", "HouseStyle", and "SaleCondition":
countplot_cols= ['MSZoning', 'HouseStyle', 'SaleCondition']

fig, axes_3= plt.subplots(nrows= 3, ncols= 1, figsize= (10, 8))
axes_3= axes_3.flatten()

for c_cols, ax_3 in zip(countplot_cols, axes_3):
  sns.countplot(data= categorical_cols, x= c_cols, ax= ax_3)
  ax_3.set_title(f'Countplot of {c_cols}')
  ax_3.set_xlabel(c_cols)
  ax_3.set_ylabel('Count')

plt.tight_layout()
plt.show()

# Identifying the 10 most frequent neighborhoods:
frequent_neighborhoods= categorical_cols['Neighborhood'].value_counts().head(10)
print(f"10 most frequent neighborhoods: \n{frequent_neighborhoods}\n")

# Calculating avg "SalePrice" of each neighborhood:
avg_saleprice_neighborhood= train_df.groupby('Neighborhood')['SalePrice'].mean()
print(f"Average SalePrice of each neighborhood: \n{round(avg_saleprice_neighborhood, 4)}\n")

# Calculating avg "SalePrice" for each "MSZoning" category:
avg_saleprice_zoning= train_df.groupby('MSZoning')['SalePrice'].mean()
print(f"Average SalePrice for each MSZoning category: \n{round(avg_saleprice_zoning, 4)}\n")

# Creating 'boxplots' comparing "SalePrice" across "Neighborhood", "MSZoning", and "HouseStyle":
boxplot_cols_2= ['Neighborhood', 'MSZoning', 'HouseStyle']

fig, axes_4= plt.subplots(nrows= 3, ncols= 1, figsize= (18, 16))
axes_4= axes_4.flatten()

for box_cols, ax_4 in zip(boxplot_cols_2, axes_4):
  sns.boxplot(data= train_df, x= box_cols, y= 'SalePrice', ax= ax_4)
  ax_4.set_title(f'Boxplot of SalePrice by {box_cols}')
  ax_4.set_xlabel(box_cols)
  ax_4.set_ylabel('SalePrice')
  ax_4.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# 6: Outlier Analysis
# Select numerical features for outlier analysis, excluding Id:
numerical_cols_outlier= numerical_cols.drop(columns= ['Id'])
outlier_cols= numerical_cols_outlier.columns
print(f"Numerical columns for outlier analysis: \n{outlier_cols}\n")

# Calculating Q1 (25th percentile) for each column:
Q1= numerical_cols_outlier.quantile(0.25)
print(f"Q1 (25th percentile) for each column: \n{Q1}\n")

# Calculating Q3 (75th percentile) for each column:
Q3= numerical_cols_outlier.quantile(0.75)
print(f"Q3 (75th percentile) for each column: \n{Q3}\n")

# Calculating IQR:
IQR= Q3 - Q1
print(f"Interquartile Range (IQR) for each column: \n{IQR}\n")

# Calculating lower bound:
lower_bound= Q1 - 1.5 * IQR
print(f"Lower bound for each column: \n{lower_bound}\n")

# Calculating upper bound:
upper_bound= Q3 + 1.5 * IQR
print(f"Upper bound for each column: \n{upper_bound}\n")

# Counting potential outlier in each numerical feature:
outlier_count= ((numerical_cols_outlier < (lower_bound)) | (numerical_cols_outlier > (upper_bound)))
print(f"Potential outlier in each numerical feature: \n{outlier_count.sum()}\n")

# Calculting percentage of potential outlier in each numerical feature:
outlier_percentage = (outlier_count.mean() * 100).round(2)
print(f"Potential outlier in each numerical feature: \n{outlier_percentage}\n")

# Displaying the 10 features with highest outlier count:
top_10_outliers_fea= outlier_count.sum().sort_values(ascending= False).head(10)
print(f"Top 10 features with highest outlier count: \n{top_10_outliers_fea}\n")

# Inspecting potential outliers in "GrLivArea", "LotArea", and "SalePrice":
fea_outlier= ['GrLivArea', 'LotArea', 'SalePrice']

for out in fea_outlier:

  flagged_rows= numerical_cols[outlier_count[out]]
  sorted_fea= flagged_rows.sort_values(by= out, ascending= False)

  if out == "SalePrice":
    cols_to_show= ['Id', 'SalePrice']
  
  else:
    cols_to_show= ['Id', out, 'SalePrice']

  print(f"Potential outliers in {out}\n"
        f"Lower bound: {lower_bound[out]} | Upper bound: {upper_bound[out]}\n"
        f"Number of flagged rows: {len(flagged_rows)}\n"
        f"{sorted_fea[cols_to_show]}\n")
  

# 7: Correlation Heatmap
# Calculating the pearson correlation matrix:
corr_matrix= numerical_cols_outlier.corr()
print(f"Pearson correlation matrix: \n{corr_matrix}\n")

# Creating a correlation heatmap using 'Seaborn':
plt.figure(figsize= (28, 24))
sns.heatmap(corr_matrix, annot= True, cmap= 'coolwarm', linewidths= 0.5)
plt.title('Correlation Heatmap')
plt.show()

# Identifying the 5 strongest positive correlation with "SalePrice":
top_5_pos_corr= corr_matrix['SalePrice'].sort_values(ascending= False).iloc[1:6]
print(f"Top 5 strongest positive correlation with SalePrice: \n{top_5_pos_corr}\n")

# Identifying the 5 strongest negative correlation with "SalePrice":
top_5_neg_corr= corr_matrix['SalePrice'].sort_values(ascending= True).head()
print(f"Top 5 strongest negative correlation with SalePrice: \n{top_5_neg_corr}\n")

# Checking for highly correlated feature pairs (absolute correlation greater than 0.80), excluding self-correlations and duplicate pairs:
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
corr_fea_pairs = upper_tri.stack().reset_index()
corr_fea_pairs.columns = ['Feature 1', 'Feature 2', 'Feature correlation']
corr_fea_pairs = corr_fea_pairs[corr_fea_pairs['Feature correlation'].abs() > 0.80]
corr_fea_pairs = corr_fea_pairs.sort_values('Feature correlation', key=abs, ascending=False)
print(f"Highly correlated feature pairs: \n{corr_fea_pairs}\n")


# 8: Target Variable Analysis (SalePrice)
# Calculating "SalePrice" - 'mean', 'median', 'standard deviation', 'minimum', and 'maximum':
stats_saleprice= train_df['SalePrice'].agg(['mean', 'median', 'std', 'min', 'max'])
print(f"SalePrice stats: \n{round(stats_saleprice, 4)}\n")

# Calculating the skewness and kurtosis of "SalePrice":
print(f"Skewness of SalePrice: {train_df['SalePrice'].skew():.4f}")
print(f"Kurtosis of SalePrice: {train_df['SalePrice'].kurt():.4f}\n")

# Applying a temporary 'log1p' transformation to "SalePrice" for comparison:
train_df['SalePrice_log'] = np.log1p(train_df['SalePrice'])

# Comparing original and log-transformed SalePrice distributions side by side:
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(train_df['SalePrice'], bins=30, kde=True, ax=axes[0])
axes[0].set_title('Original SalePrice')
axes[0].set_xlabel('SalePrice')
axes[0].set_ylabel('Frequency')

sns.histplot(train_df['SalePrice_log'], bins=30, kde=True, ax=axes[1], color='darkorange')
axes[1].set_title('Log-transformed SalePrice')
axes[1].set_xlabel('log1p(SalePrice)')
axes[1].set_ylabel('Frequency')

plt.suptitle('Comparison of SalePrice Distributions', fontsize=14)
plt.tight_layout()
plt.show()

# Calculating skewness after the log transformation:
print(f"Skewness of log-transformed SalePrice: {train_df['SalePrice_log'].skew():.4f}")

# Removing the temporary column so it can't leak into the feature matrix later:
train_df.drop(columns='SalePrice_log', inplace=True)

# "SalePrice" Distribution Observation:
# Observation 1: Mean is higher that median.
# Observation 2: "SalePrice" is highly right skew (Skewness of SalePrice: 1.8829).
# Observation 3: Some houses have unusually high sale prices.
# Observation 4: Log transformation reduces the positive skewness.
# Observation 5: Most houses are concentrated in the lower-to-middle price range.


# 9: Data Quality Checks
