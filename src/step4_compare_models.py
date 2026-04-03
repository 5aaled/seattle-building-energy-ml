"""
Step 4: Comparison of different supervised models
Seattle Building Energy Benchmarking - Non-Residential Buildings
Compares LinearRegression, DecisionTreeRegressor, SVR
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X_PATH = os.path.join(PROJECT_ROOT, "data", "prepared", "X_prepared.csv")
Y_PATH = os.path.join(PROJECT_ROOT, "data", "prepared", "y.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV = 5

SCORING = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]


print("=" * 60)
print("STEP 4: COMPARISON OF SUPERVISED MODELS")
print("=" * 60)

X = pd.read_csv(X_PATH)
y = pd.read_csv(Y_PATH)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")





X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"Train: {len(X_train)} samples")
print(f"Test: {len(X_test)} samples")


print("\n--- LinearRegression ---")

pipeline_linear = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression()),
])

cv_linear = cross_validate(pipeline_linear, X_train, y_train, cv=CV, scoring=SCORING)
pipeline_linear.fit(X_train, y_train)
y_train_pred_linear = pipeline_linear.predict(X_train)
y_test_pred_linear = pipeline_linear.predict(X_test)

metrics_linear = {
    "model": "LinearRegression",
    "train_r2": r2_score(y_train, y_train_pred_linear),
    "test_r2": r2_score(y_test, y_test_pred_linear),
    "train_mae": mean_absolute_error(y_train, y_train_pred_linear),
    "test_mae": mean_absolute_error(y_test, y_test_pred_linear),
    "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred_linear)),
    "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred_linear)),
    "cv_r2_mean": cv_linear["test_r2"].mean(),
    "cv_r2_std": cv_linear["test_r2"].std(),
}


print("DecisionTreeRegressor ---")

pipeline_dt = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DecisionTreeRegressor(random_state=RANDOM_STATE)),
])

cv_dt = cross_validate(pipeline_dt, X_train, y_train, cv=CV, scoring=SCORING)
pipeline_dt.fit(X_train, y_train)
y_train_pred_dt = pipeline_dt.predict(X_train)
y_test_pred_dt = pipeline_dt.predict(X_test)

metrics_dt = {
    "model": "DecisionTreeRegressor",
    "train_r2": r2_score(y_train, y_train_pred_dt),
    "test_r2": r2_score(y_test, y_test_pred_dt),
    "train_mae": mean_absolute_error(y_train, y_train_pred_dt),
    "test_mae": mean_absolute_error(y_test, y_test_pred_dt),
    "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred_dt)),
    "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred_dt)),
    "cv_r2_mean": cv_dt["test_r2"].mean(),
    "cv_r2_std": cv_dt["test_r2"].std(),
}


print("--- SVR ---")

pipeline_svr = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", SVR()),
])

cv_svr = cross_validate(pipeline_svr, X_train, y_train, cv=CV, scoring=SCORING)
pipeline_svr.fit(X_train, y_train)
y_train_pred_svr = pipeline_svr.predict(X_train)
y_test_pred_svr = pipeline_svr.predict(X_test)

metrics_svr = {
    "model": "SVR",
    "train_r2": r2_score(y_train, y_train_pred_svr),
    "test_r2": r2_score(y_test, y_test_pred_svr),
    "train_mae": mean_absolute_error(y_train, y_train_pred_svr),
    "test_mae": mean_absolute_error(y_test, y_test_pred_svr),
    "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred_svr)),
    "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred_svr)),
    "cv_r2_mean": cv_svr["test_r2"].mean(),
    "cv_r2_std": cv_svr["test_r2"].std(),
}


print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

results = [metrics_linear, metrics_dt, metrics_svr]
print("here is the result ")

results_df = pd.DataFrame(results)

print(results_df)

results_df.to_csv(
    os.path.join(PROJECT_ROOT, "data", "prepared", "model_comparison_results.csv"),
    index=False,
)
print(f"\nResults saved to: data/prepared/model_comparison_results.csv")

print("\n" + "=" * 60)
print("STEP 4 COMPLETE: Model Comparison Done")

