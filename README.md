# House Price Prediction

## Getting Started

### Prerequisites

- Python 3.12
- Git
- VS Code (optional)

### Installation

**1. Clone the repository (Windows / macOS / Linux):**

```bash
git clone https://github.com/tyaginishant0071-hash/House-Price-Prediction.git
```

**2. Navigate to the project directory:**

```bash
cd House-Price-Prediction
```

**3. Create a virtual environment:**

**Windows (PowerShell / CMD):**

```powershell
py -3.12 -m venv venv
```

**macOS / Linux:**

```bash
python3.12 -m venv venv
```

*Ensure Python 3.12 is installed and accessible from your terminal.*

**4. Activate the virtual environment:**

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
venv\Scripts\activate.bat
```

**macOS / Linux:**

```bash
source venv/bin/activate
```

**5. Install dependencies (all operating systems):**

```bash
python -m pip install -r requirements.txt
```

**6. Verify installed dependencies:**

```bash
python -m pip check
```

Expected output:

```text
No broken requirements found.
```

### Dataset

Download the dataset from Kaggle:

[House Prices — Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)

**Dataset setup:**

1. Download `train.csv` and `test.csv`.
2. Create the `data/raw/` directory if it does not exist.
3. Place both CSV files inside `data/raw/`.
4. Create the `data/processed/` directory if it does not exist.

**Expected directory structure:**

```text
House-Price-Prediction/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── requirements.txt
└── README.md
```

**Note:** Raw and processed datasets are excluded from Git tracking through `.gitignore`. Download the original dataset from Kaggle before running the project.


## Exploratory Data Analysis

- The data contains 1,460 training rows and 1,459 test rows. `SalePrice` is available only in training.
- Missing values account for 6.62% of training cells and 6.75% of test cells. The same six columns exceed 40% missingness in both datasets.
- Overall quality and above-ground living area had the strongest positive correlations with sale price. Applying `log1p` reduced target skewness from 1.88 to 0.12.
- The IQR rule flagged 1,657 cells across 897 training rows. Both datasets had zero duplicate rows and zero constant columns, and all training sale prices were positive.

See the [full EDA report](reports/eda_summary.md) for methods, detailed findings, and planned preprocessing.

## Data Cleaning

- Split the labeled data into 80% training (1,168 rows) and 20% validation (292 rows) using `random_state=42`. The 1,459 test rows were cleaned separately.
- Filled missing categorical values with `Unknown` and used medians calculated from the training split to fill missing numerical values in all three sets.
- Verified that each cleaned feature set has 79 columns and zero missing values. Saved the cleaned features, training and validation targets, and test IDs in `data/processed/`.

See the [full data cleaning report](reports/data_cleaning_summary.md) for details and the list of saved files.

## Feature Engineering

- Added five numerical features across the training, validation, and test sets: `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, and `TotalPorchSF`.
- Set four negative engineered age values to zero. All three datasets have the same 84 columns in the same order, with zero missing values and zero negative engineered ages.
- Saved the engineered feature sets as separate CSVs in `data/processed/`.

See the [full feature engineering report](reports/feature_engineering_summary.md) for formulas, verification results, and output filenames.

## Data Preprocessing

- One-hot encoded 43 categorical features into 265 numerical columns and scaled the 41 original numerical features using `StandardScaler`. Both transformers were fitted on training data only and applied to training, validation, and test data.
- The final datasets contain 306 numerical features with identical column order, zero missing values, and zero duplicate column names. Row counts are 1,168 for training, 292 for validation, and 1,459 for test.
- Saved the preprocessed CSVs in `data/processed/` and the fitted transformers and feature-column lists in `models/`. The datasets are ready for model training and evaluation.

See the [full data preprocessing report](reports/data_preprocessing_summary.md) for details and saved filenames.

## Model Training and Evaluation

- Trained Linear Regression, Random Forest, and Gradient Boosting models using 1,168 training rows with 306 features, then evaluated them on 292 validation rows using MAE, RMSE, and R².
- Gradient Boosting achieved the best validation performance, with MAE of 16,360.67, RMSE of 26,772.24, and R² of 0.9066.
- Selected Gradient Boosting for further hyperparameter tuning in the Model Optimization phase.

See the [full model training report](reports/model_training_summary.md) for model parameters and comparison results.

## Model Optimization

- Tuned Gradient Boosting using GridSearchCV with 5-fold cross-validation across eight parameter combinations. The best settings were `n_estimators=200`, `learning_rate=0.1`, and `max_depth=3`.
- The tuned model achieved validation MAE of 15,864.42, RMSE of 26,239.98, and R² of 0.9102, improving all three metrics over the original model.
- Saved the final model as `models/gradient_boosting_tuned.joblib` for generating test predictions.

See the [full model optimization report](reports/model_optimization_summary.md) for tuning details and performance comparisons.

## Final Predictions

- Generated house-price predictions for all 1,459 test properties using the tuned Gradient Boosting model.
- Verified prediction values and original ID order, with zero missing, infinite, or nonpositive sale prices.
- Saved `data/processed/submission.csv` with `Id` and `SalePrice` columns, rounding prices to four decimal places. The submission file was successfully uploaded to Kaggle and achieved a public leaderboard RMSLE score of 0.13251.

See the [full final prediction report](reports/final_prediction_summary.md) for validation details.

### Kaggle Submission Results

- **Competition:** House Prices — Advanced Regression Techniques
- **Final Model:** Tuned Gradient Boosting Regressor
- **Submission File:** `data/processed/submission.csv`
- **Submission Status:** Successfully Completed
- **Public Leaderboard Score (RMSLE):** 0.13251
- **Total Test Predictions:** 1,459

The tuned Gradient Boosting model successfully generated predictions for all Kaggle test records.

The submission was accepted and evaluated by Kaggle, achieving a public leaderboard RMSLE score of **0.13251**.
