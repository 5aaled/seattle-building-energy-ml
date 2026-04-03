import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

# =============================================================================
# CONFIGURATION
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
X_PATH = os.path.join(PROJECT_ROOT, "data", "prepared", "X_prepared.csv")
Y_PATH = os.path.join(PROJECT_ROOT, "data", "prepared", "y.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "data", "figures")
MODELS_DIR = os.path.join(PROJECT_ROOT, "data", "prepared")

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV = 5


X = pd.read_csv(X_PATH)
y = pd.read_csv(Y_PATH).squeeze()

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

print("\n--- Train-Test Split ---")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", DecisionTreeRegressor(random_state=RANDOM_STATE)),
])

param_grid = {
    "model__max_depth": [5, 10, 15, 20, None],
    "model__min_samples_leaf": [2, 5, 10, 20],
    "model__min_samples_split": [2, 5, 10],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=CV,
    scoring="r2",
    n_jobs=-1,
    verbose=1,
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV R2: {grid_search.best_score_:.4f}")

# =============================================================================
# 4. EVALUATE BEST MODEL
# =============================================================================
print("\n--- Best Model Performance ---")

best_pipeline = grid_search.best_estimator_
y_train_pred = best_pipeline.predict(X_train)
y_test_pred = best_pipeline.predict(X_test)

print(f"Train R2:  {r2_score(y_train, y_train_pred):.4f}")
print(f"Test R2:   {r2_score(y_test, y_test_pred):.4f}")
print(f"Train MAE: {mean_absolute_error(y_train, y_train_pred):.2f}")
print(f"Test MAE:  {mean_absolute_error(y_test, y_test_pred):.2f}")
print(f"Train RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.2f}")
print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")

# =============================================================================
# 5. FEATURE IMPORTANCE
# =============================================================================
print("\n--- Feature Importance ---")

# Extract the fitted DecisionTree from the pipeline
dt_model = best_pipeline.named_steps["model"]
feature_importances = dt_model.feature_importances_
feature_names = X.columns.tolist()

# Sort by importance (descending)
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": feature_importances,
}).sort_values("importance", ascending=True)

print("\nTop 10 most important features:")
for _, row in importance_df.tail(10).iloc[::-1].iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# =============================================================================
# 6. FEATURE IMPORTANCE PLOT (histogram)
# =============================================================================
print("\n--- Saving feature importance plot ---")

os.makedirs(FIGURES_DIR, exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))
bars = ax.barh(importance_df["feature"], importance_df["importance"], color=colors)
ax.set_xlabel("Feature Importance")
ax.set_ylabel("Feature")
ax.set_title("DecisionTreeRegressor - Feature Importance (optimized model)")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "step5_feature_importance.png"), dpi=100, bbox_inches="tight")
plt.close()

print(f"  Saved: {FIGURES_DIR}/step5_feature_importance.png")

print("\n--- Saving results ---")

# Save best params and metrics
results = {
    "best_params": str(grid_search.best_params_),
    "best_cv_r2": grid_search.best_score_,
    "test_r2": r2_score(y_test, y_test_pred),
    "test_mae": mean_absolute_error(y_test, y_test_pred),
    "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
}
pd.DataFrame([results]).to_csv(
    os.path.join(MODELS_DIR, "step5_optimization_results.csv"),
    index=False,
)

# Save feature importance
importance_df.to_csv(
    os.path.join(MODELS_DIR, "step5_feature_importance.csv"),
    index=False,
)

print(f"  Saved: {MODELS_DIR}/step5_optimization_results.csv")
print(f"  Saved: {MODELS_DIR}/step5_feature_importance.csv")

print("\n" + "=" * 60)
print("STEP 5 COMPLETE: Optimization and Interpretation Done")
print("=" * 60)