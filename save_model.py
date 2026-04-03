"""
Save the trained model with BentoML (Mission Part 2).
Requires: data/prepared/X_prepared.csv, y.csv, preprocessor.joblib (from Step 3).
"""

import os
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import bentoml

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PREPARED_DIR = os.path.join(PROJECT_ROOT, "data", "prepared")

RANDOM_STATE = 42

BEST_PARAMS = {
    "max_depth": 5,
    "min_samples_leaf": 5,
    "min_samples_split": 2,
}

X_PATH = os.path.join(PREPARED_DIR, "X_prepared.csv")
Y_PATH = os.path.join(PREPARED_DIR, "y.csv")
PREPROCESSOR_PATH = os.path.join(PREPARED_DIR, "preprocessor.joblib")


def main():
    if not os.path.isfile(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            f"Missing {PREPROCESSOR_PATH}. Run: python src/step3_prepare_features.py"
        )

    X_prepared = pd.read_csv(X_PATH)
    y = pd.read_csv(Y_PATH).squeeze()

    preprocessor = joblib.load(PREPROCESSOR_PATH)

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(random_state=RANDOM_STATE, **BEST_PARAMS)),
        ]
    )
    pipeline.fit(X_prepared, y)

    bentoml.sklearn.save_model(
        "seattle_energy_model",
        pipeline,
        custom_objects={
            "preprocessor": preprocessor,
            "feature_names": X_prepared.columns.tolist(),
            "default_values": X_prepared.median().to_dict(),
        },
    )
    print("Model saved successfully.")


if __name__ == "__main__":
    main()
