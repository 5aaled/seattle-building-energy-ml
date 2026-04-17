"""
Step 3: Preparation of features for modeling
Seattle Building Energy Benchmarking - Non-Residential Buildings
Prepares X (features) and y (target) for supervised modeling
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer     # It lets you apply different transformations to different columns.
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "enriched", "building_consumption_enriched.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "prepared")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "data", "figures")

TARGET_COLUMN = "SiteEUI(kBtu/sf)"


print("STEP 3: PREPARATION OF FEATURES FOR MODELING")


df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} rows from enriched data")
initial_cols = len(df.columns)



cols_dropped= [ "OSEBuildingID", "DataYear", "PropertyName", "Address", "City", "State", "TaxParcelIdentificationNumber", "ZipCode","ListOfAllPropertyUseTypes", "LargestPropertyUseType","SecondLargestPropertyUseType", "SecondLargestPropertyUseTypeGFA","ThirdLargestPropertyUseType", "ThirdLargestPropertyUseTypeGFA", "YearsENERGYSTARCertified", "SiteEUIWN(kBtu/sf)", "SourceEUI(kBtu/sf)", "SourceEUIWN(kBtu/sf)", "SiteEnergyUse(kBtu)", "SiteEnergyUseWN(kBtu)", "SteamUse(kBtu)", "Electricity(kWh)", "Electricity(kBtu)", "NaturalGas(therms)", "NaturalGas(kBtu)","DefaultData", "Comments", "ComplianceStatus", "Outlier", "TotalGHGEmissions", "GHGEmissionsIntensity", "PropertyGFATotal", "PropertyGFAParking", "PropertyGFABuilding(s)","YearBuilt", "PrimaryPropertyType", "Neighborhood",]


df = df.drop(columns=cols_dropped)
print(f"  Dropped {len(cols_dropped)} columns. Remaining: {len(df.columns)}")


print("\n Target distribution and outlier handling ")

# Plot target distribution
os.makedirs(FIGURES_DIR, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df[TARGET_COLUMN], bins=50, edgecolor="black", alpha=0.7)
ax.set_title(f"Distribution of {TARGET_COLUMN} (before outlier removal)")
ax.set_xlabel(TARGET_COLUMN)
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "step3_target_distribution.png"), dpi=100, bbox_inches="tight")
plt.close()
print("  Saved: step3_target_distribution.png")

# Quantile-based outlier removal (keep 1st-99th percentile to avoid losing too many rows)
low_q = df[TARGET_COLUMN].quantile(0.01)
high_q = df[TARGET_COLUMN].quantile(0.99)
rows_before = len(df)
df = df[(df[TARGET_COLUMN] >= low_q) & (df[TARGET_COLUMN] <= high_q)]
rows_removed = rows_before - len(df)
print(f"  Outlier removal (1st-99th percentile): kept {len(df)} rows, removed {rows_removed}")

# Plot target distribution AFTER outlier removal
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df[TARGET_COLUMN], bins=50, edgecolor="black", alpha=0.7, color="steelblue")
ax.set_title(f"Distribution of {TARGET_COLUMN} (after outlier removal)")
ax.set_xlabel(TARGET_COLUMN)
ax.set_ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "step3_target_distribution_after.png"), dpi=100, bbox_inches="tight")
plt.close()
print("  Saved: step3_target_distribution_after.png")




print("\n Correlation matrix (redundant features)  ")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if TARGET_COLUMN in numeric_cols:
    numeric_cols.remove(TARGET_COLUMN)

corr_matrix = df[numeric_cols].corr()
to_drop = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.9:
            col_i = corr_matrix.columns[i]
            col_j = corr_matrix.columns[j]
            # Drop the one with fewer unique values or keep BuildingAge over BuildingAgeSquared
            if col_i == "BuildingAge" and col_j == "BuildingAgeSquared":
                to_drop.add("BuildingAgeSquared")  # Keep BuildingAge, drop squared
            elif col_i in df.columns and col_j in df.columns:
                to_drop.add(col_j)

df = df.drop(columns=to_drop, errors="ignore")
if to_drop:
    print(f"  Dropped highly correlated: {list(to_drop)}")
else:
    print("  No highly correlated pairs (>0.9) to remove")

# Plot correlation matrix - get numeric cols from current df (includes target)
plot_cols = df.select_dtypes(include=[np.number]).columns.tolist()
corr_cols = ([TARGET_COLUMN] + [c for c in plot_cols if c != TARGET_COLUMN])[:12]
corr_plot = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_plot, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
plt.title("Correlation Matrix (key features)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "step3_correlation_matrix.png"), dpi=100, bbox_inches="tight")
plt.close()
print("  Saved: step3_correlation_matrix.png")

# =============================================================================
# 5. FEATURE-TARGET PLOTS
# =============================================================================
print("\n--- Feature-target relationship plots ---")

# Boxplot: Target vs PrimaryPropertyTypeGrouped
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(data=df, x="PrimaryPropertyTypeGrouped", y=TARGET_COLUMN)
plt.xticks(rotation=45, ha="right")
plt.title(f"{TARGET_COLUMN} by Property Type (grouped)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "step3_target_vs_property_type.png"), dpi=100, bbox_inches="tight")
plt.close()
print("  Saved: step3_target_vs_property_type.png")

# Scatter: Target vs BuildingAge
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["BuildingAge"], df[TARGET_COLUMN], alpha=0.4, s=15)
ax.set_xlabel("Building Age")
ax.set_ylabel(TARGET_COLUMN)
ax.set_title(f"{TARGET_COLUMN} vs Building Age")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "step3_target_vs_building_age.png"), dpi=100, bbox_inches="tight")
plt.close()
print("  Saved: step3_target_vs_building_age.png")

# Scatter: Target vs LogPropertyGFATotal
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["LogPropertyGFATotal"], df[TARGET_COLUMN], alpha=0.4, s=15)
ax.set_xlabel("Log(Property GFA Total)")
ax.set_ylabel(TARGET_COLUMN)
ax.set_title(f"{TARGET_COLUMN} vs Building Size (log)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "step3_target_vs_log_gfa.png"), dpi=100, bbox_inches="tight")
plt.close()
print("  Saved: step3_target_vs_log_gfa.png")

# =============================================================================
# 6. SEPARATE X AND y
# =============================================================================
print("\n--- Separating X and y ---")

y = df[TARGET_COLUMN].copy()
X = df.drop(columns=[TARGET_COLUMN])

print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")


print("\n--- Encoding categorical features ---")

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"  Categorical columns: {categorical_cols}")
print(f"  Numeric columns: {len(numeric_cols)}")

if categorical_cols:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore", max_categories=15), categorical_cols),
        ],
        remainder="drop",
    )
    X_encoded = preprocessor.fit_transform(X)  #X_encoded is: Fully numeric But does NOT have column names
    feature_names = numeric_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols))
    X_prepared = pd.DataFrame(X_encoded, columns=feature_names, index=X.index)
else:
    preprocessor = ColumnTransformer(
        transformers=[("num", "passthrough", numeric_cols)],
        remainder="drop",
    )
    X_encoded = preprocessor.fit_transform(X)
    X_prepared = pd.DataFrame(X_encoded, columns=numeric_cols, index=X.index)

print(f"  X_prepared shape after encoding: {X_prepared.shape}")



os.makedirs(OUTPUT_DIR, exist_ok=True)
X_prepared.to_csv(os.path.join(OUTPUT_DIR, "X_prepared.csv"), index=False)
y.to_csv(os.path.join(OUTPUT_DIR, "y.csv"), index=False)

prep_path = os.path.join(OUTPUT_DIR, "preprocessor.joblib")
joblib.dump(preprocessor, prep_path)
print(f"  Saved: {prep_path} (for save_model.py / API — same encoder as X_prepared)")




