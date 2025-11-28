# =========================================
# HOUSE PRICES - FINAL PIPELINE + SUBMISSION
# =========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Try to use XGBoost if available (usually gives a better score)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ----- STEP 1: LOAD DATA -----

def load_house_data(train_path: str = "train.csv",
                    test_path: str = "test.csv"):
    """
    Load the Kaggle House Prices train and test CSVs.

    Returns:
        train_df: training DataFrame (has SalePrice)
        test_df: test DataFrame (no SalePrice)
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)
    print("\nTrain columns:")
    print(train_df.columns.tolist()[:20], "...")  # show first 20 column names

    return train_df, test_df


def explore_numeric_features(train_df: pd.DataFrame):
    """
    Inspect numeric columns and see which ones are most related to SalePrice.
    This is optional / just for intuition.
    """
    num_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    print("\nTotal numeric columns:", len(num_cols))
    print("Some numeric columns:", num_cols[:15], "...")

    corr_series = train_df[num_cols].corr()["SalePrice"].sort_values(
        ascending=False
    )

    print("\nTop 10 features most positively correlated with SalePrice:")
    print(corr_series.head(10))

    print("\nTop 10 features most negatively correlated with SalePrice:")
    print(corr_series.tail(10))


# ----- STEP 2: BUILD PREPROCESSOR + MODEL -----

def build_preprocessor(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Build a ColumnTransformer that:
      - imputes numeric with median + scales
      - imputes categorical with most_frequent + one-hot encodes
    """
    # Drop target and keep only features
    X = train_df.drop(columns=["SalePrice"])
    X_test = test_df.copy()

    # Identify numeric / categorical columns
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # It's usually better not to use Id as a predictive feature
    if "Id" in numeric_features:
        numeric_features.remove("Id")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, X, X_test


def build_model():
    """
    Build the regression model.
    Uses XGBRegressor if available, otherwise RandomForestRegressor.
    """
    if HAS_XGB:
        print("Using XGBRegressor as the model.")
        model = XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist",
        )
    else:
        print("xgboost not installed, using RandomForestRegressor instead.")
        model = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        )
    return model


# ----- STEP 3: TRAIN, CV, AND PREDICT -----

def cross_validate_model(preprocessor, model, X, y_log):
    """
    Run K-Fold cross-validation and print average RMSE on log(SalePrice).
    """
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # scoring is negative RMSE because sklearn returns "higher is better"
    scores = cross_val_score(
        pipeline,
        X,
        y_log,
        cv=kf,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    rmse_scores = -scores
    print("\nCV RMSE (on log1p(SalePrice)):")
    print("Fold scores:", rmse_scores)
    print("Mean RMSE:", rmse_scores.mean())
    print("Std RMSE:", rmse_scores.std())

    return pipeline


def fit_full_and_predict(pipeline, X, y_log, X_test):
    """
    Fit the pipeline on full training data and predict on the test set.
    Returns predictions in original SalePrice scale.
    """
    print("\nFitting model on full training data...")
    pipeline.fit(X, y_log)

    print("Predicting on test data...")
    test_preds_log = pipeline.predict(X_test)
    test_preds = np.expm1(test_preds_log)  # invert log1p

    return test_preds


def save_submission(test_df: pd.DataFrame, predictions: np.ndarray,
                    filename: str = "submission.csv"):
    """
    Save Kaggle submission file with columns: Id, SalePrice
    """
    submission = pd.DataFrame(
        {
            "Id": test_df["Id"],
            "SalePrice": predictions,
        }
    )
    submission.to_csv(filename, index=False)
    print(f"\nSubmission file saved as: {filename}")
    print(submission.head())


# ----- MAIN -----

def main():
    # 1) Load data
    train_df, test_df = load_house_data("train.csv", "test.csv")

    # Optional: quick exploration
    print("\nSample of train data (SalePrice + a few features):")
    print(
        train_df[["Id", "SalePrice", "OverallQual", "GrLivArea", "YearBuilt"]]
        .head()
    )

    # explore_numeric_features(train_df)  # uncomment if you want to see correlations

    # 2) Build preprocessor + split features/target
    preprocessor, X, X_test = build_preprocessor(train_df, test_df)

    # Target: log1p of SalePrice (common trick for this competition)
    y = train_df["SalePrice"].values
    y_log = np.log1p(y)

    # 3) Build model
    model = build_model()

    # 4) Cross-validate (to see expected performance)
    pipeline = cross_validate_model(preprocessor, model, X, y_log)

    # 5) Fit on full data and predict for test
    test_preds = fit_full_and_predict(pipeline, X, y_log, X_test)

    # 6) Save submission CSV
    save_submission(test_df, test_preds, filename="submission_Final.csv")


if __name__ == "__main__":
    main()
