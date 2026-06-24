# 🏠 Handling Missing Categorical Data: The "Missing" Category Trick

A short, focused notebook demonstrating how to handle missing values in categorical features using the **"Missing" category** technique with `SimpleImputer`, on the House Prices dataset.

## 🔍 What this notebook covers

- Identifying missing value percentages in categorical columns (`GarageQual`, `FireplaceQu`)
- Visualizing category distributions with bar plots
- Treating missing values as their own valid category (`'Missing'`) instead of dropping rows or guessing a value
- Correctly applying `SimpleImputer(strategy='constant', fill_value='Missing')` **after** the train/test split, to avoid data leakage

## 🧠 Why this approach

For categorical features, a missing value often isn't random — it can be informative on its own (e.g., no `FireplaceQu` likely means no fireplace exists at all). Rather than imputing with the mode or dropping the column, treating "missing" as its own category preserves that signal and lets downstream models learn from it directly.

## 📊 Dataset

This notebook uses the [House Prices dataset](https://www.kaggle.com/datasets), specifically:
- `GarageQual` — Garage quality (categorical, ~5.5% missing)
- `FireplaceQu` — Fireplace quality (categorical, ~47.3% missing)
- `SalePrice` — Target variable (no missing values)

## 🛠️ Tech stack

- Python 3
- pandas, numpy
- matplotlib
- scikit-learn (`train_test_split`, `SimpleImputer`)

## 📁 Notebook structure

| Step | What it does |
|---|---|
| 1. Load data | Reads selected columns from the housing dataset |
| 2. Check missing values | Calculates percentage of nulls per column |
| 3. Visualize category counts | Bar plot of `GarageQual` value distribution |
| 4. Train/test split | Splits data **before** imputation to prevent leakage |
| 5. Impute missing values | Fills missing categorical values with `'Missing'` using `SimpleImputer` |

## ✅ Key takeaway

> Always split your data into train and test sets **before** fitting an imputer. Fitting on the full dataset first leaks test set information into training — the same principle that applies to scaling also applies to imputation.

## 🚀 How to run

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install pandas numpy matplotlib scikit-learn
jupyter notebook missing-categorical-data.ipynb
```

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
