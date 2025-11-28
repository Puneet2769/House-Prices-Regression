# 🏠 House Prices — Final Pipeline & Submission  
### Clean end-to-end pipeline for the Kaggle "House Prices: Advanced Regression Techniques" competition

A robust, production-style pipeline that loads data, preprocesses numeric & categorical features, runs cross-validation, fits a strong regressor (XGBoost if available, otherwise RandomForest), and generates a Kaggle-ready submission.

---

## 📘 Competition  
**Kaggle:** https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

**Dataset (expected files):**  
- `train.csv` → training set with `SalePrice` (target)  
- `test.csv` → test set (no `SalePrice`)  
- Keep the CSVs in the project root (dataset not included here)

---

## ⚙️ What this Project Does (high level)

1. Loads `train.csv` and `test.csv` and prints basic shapes & sample columns.  
2. Builds a `ColumnTransformer` that:
   - imputes numeric features with median + standard-scales them, and
   - imputes categorical features with most frequent value + one-hot encodes (ignore unknowns).
3. Uses `log1p(SalePrice)` as the target (common trick for this competition).  
4. Trains a model:
   - Uses **XGBRegressor** if `xgboost` is installed, otherwise falls back to **RandomForestRegressor**.
5. Runs 5-fold CV (shuffled KFold) and reports RMSE on `log1p(SalePrice)`.  
6. Fits the pipeline on full training data, predicts test set, inverts `log1p`, and saves submission CSV.

---

## 🧠 Model & Preprocessing Details

- **Preprocessor**
  - Numeric: `SimpleImputer(strategy="median")` → `StandardScaler()`
  - Categorical: `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")`

- **Model**
  - Preferred: `XGBRegressor` (if installed) with sensible defaults (learning_rate=0.05, n_estimators=1000, max_depth=4, subsample/colsample tuning).
  - Fallback: `RandomForestRegressor` (n_estimators=400, n_jobs=-1).
  - Target is `np.log1p(SalePrice)` during training; predictions are `np.expm1()`-ed before saving.

---

## 🚀 Quick Start — Run locally

Make sure these files are present in project root:

train.csv
test.csv
house_prices_pipeline.py # your script
requirements.txt

Run:

```bash
python house_prices_pipeline.py
```
What the script does:

Prints data shapes and a small train sample.

Runs 5-fold CV and prints RMSE (on log1p scale).

Trains on all training data and writes submission_Final.csv with columns Id, SalePrice.

📁 Repository structure
bash
Copy code
├── house_prices_pipeline.py   # main pipeline (the code you shared)
├── submission_Final.csv       # generated submission (after running)
├── train.csv                  # (not included)
├── test.csv                   # (not included)
├── requirements.txt
└── README.md
# ⚠️ Notes & Tips
Id is excluded from numeric features by default — it’s not used as a predictor.

If you want higher leaderboard performance:

Add targeted feature engineering (MSSubClass, neighborhood encodings, interaction terms).

Try more advanced imputations for certain columns (e.g., LotFrontage by neighborhood).

Use feature selection or target-encoded categorical features (with CV-safe techniques).

Blend XGBoost with LightGBM / CatBoost for robust ensembles.

CV uses neg_root_mean_squared_error (sklearn convention), and printed RMSE is the positive value.


# 👤 Author
Puneet Poddar
Kaggle: https://www.kaggle.com/puneet2769

# 📌 License / Attribution
Feel free to reuse or adapt this pipeline for your experiments. If you publish results, credit the original data source (Kaggle competition).
