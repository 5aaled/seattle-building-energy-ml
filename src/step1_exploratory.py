

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(PROJECT_ROOT)
CSV_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "2016_Building_Energy_Benchmarking.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "data", "figures")
CLEAN_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "clean", "building_consumption_clean.csv")



df = pd.read_csv(CSV_PATH)
initial_count = len(df)
print(f"Initial number of rows: {initial_count}")
print(f"Initial number of columns: {len(df.columns)}")

# 2. FILTER: NON-RESIDENTIAL BUILDINGS ONLY

RESIDENTIAL_TYPES = ["Multifamily LR (1-4)", "Multifamily MR (5-9)", "Multifamily HR (10+)"]
df = df[~df["BuildingType"].isin(RESIDENTIAL_TYPES)].copy()
after_residential_filter = len(df)
print(f"\nAfter filtering to NON-RESIDENTIAL only: {after_residential_filter} rows")
print(f"  (Excluded: {initial_count - after_residential_filter} residential buildings)")


print("\n" + "=" * 60)
print("DATA QUALITY: IDENTIFYING ABERRANT BUILDINGS")
print("=" * 60)

default_data_count = (df["DefaultData"] == True).sum()
print(f"\nDefaultData=True (estimated values): {default_data_count} rows -> EXCLUDE")
print("\nComplianceStatus distribution:")
print(df["ComplianceStatus"].value_counts())
non_compliant = df[df["ComplianceStatus"] != "Compliant"]
print(f"Non-Compliant rows to exclude: {len(non_compliant)}")
gfa_zero = (df["PropertyGFATotal"] == 0) | (df["PropertyGFATotal"].isna())
print(f"\nPropertyGFATotal = 0 or missing: {gfa_zero.sum()} rows")
print("\nOutlier column:")
print(df["Outlier"].value_counts(dropna=False))
invalid_year = (df["YearBuilt"] > 2016) | (df["YearBuilt"] < 1800)
print(f"\nYearBuilt invalid (<1800 or >2016): {invalid_year.sum()} rows")

# =============================================================================
# 4. TARGET SELECTION & RECOMMENDATION
# =============================================================================
print("\n" + "=" * 60)
print("TARGET VARIABLE RECOMMENDATION")
print("=" * 60)
#  energy and CO2, either total or per sq ft
target_candidates = {
    "SiteEUI(kBtu/sf)": "Energy use per sq ft",
    "SiteEnergyUse(kBtu)": "Total energy consumption",
    "TotalGHGEmissions": "Total CO2 emissions (metric tons)",
    "GHGEmissionsIntensity": "CO2 emissions per sq ft",
}
print("\nCandidate targets for modeling:")
for col, desc in target_candidates.items():
    valid = df[df[col].notna() & (df[col] > 0)]
    skew = valid[col].skew() if len(valid) > 0 else np.nan
    print(f"  - {col}: {len(valid)} valid | skew={skew:.2f} | {desc}")


# skew refers to skewness, a statistical measure that tells you how a distribution is shaped.
#
# skew = 0 → perfectly symmetric distribution
# skew > 0 → right-skewed (long tail on the right, many small values, few large ones)
# skew < 0 → left-skewed (long tail on the left, many large values, few small ones)

TARGET_COLUMN = "SiteEUI(kBtu/sf)"
print(f"CHOSEN TARGET: {TARGET_COLUMN}")
target_zero = (df[TARGET_COLUMN] == 0) | (df[TARGET_COLUMN].isna())
print(f"\nTarget ({TARGET_COLUMN}) = 0 or missing: {target_zero.sum()} rows -> EXCLUDE")

# =============================================================================
# 5. APPLY FILTERS
# =============================================================================
print("\n" + "=" * 60)
print("APPLYING CLEANING FILTERS")
print("=" * 60)

df_clean = df.copy()
# Fix negative electricity (data error: convert to positive when building uses energy)
df_clean.loc[df_clean["Electricity(kWh)"] < 0, "Electricity(kWh)"] = df_clean.loc[df_clean["Electricity(kWh)"] < 0, "Electricity(kWh)"].abs()
df_clean.loc[df_clean["Electricity(kBtu)"] < 0, "Electricity(kBtu)"] = df_clean.loc[df_clean["Electricity(kBtu)"] < 0, "Electricity(kBtu)"].abs()

# 1. Data quality
# Compliant → data was checked and meets standards.
# Non-Compliant → data has known problems.
# Error - Correct Default Data → values are estimated, not measured.
# Missing Data → key fields are missing.
df_clean = df_clean[df_clean["ComplianceStatus"] == "Compliant"]
print(f"After ComplianceStatus=Compliant: {len(df_clean)} rows")
df_clean = df_clean[df_clean[TARGET_COLUMN].notna() & (df_clean[TARGET_COLUMN] > 0)]
print(f"After target valid (>0): {len(df_clean)} rows")
df_clean = df_clean[df_clean["PropertyGFATotal"].notna() & (df_clean["PropertyGFATotal"] > 0)]
print(f"After PropertyGFATotal valid (>0): {len(df_clean)} rows")
df_clean = df_clean[(df_clean["YearBuilt"] >= 1800) & (df_clean["YearBuilt"] <= 2016)]
print(f"After YearBuilt valid (1800-2016): {len(df_clean)} rows")

GFA_MAX_SQFT = 8_000_000
before_gfa_cap_count = len(df_clean)
df_before_gfa_cap = df_clean.copy()
df_clean = df_clean[df_clean["PropertyGFATotal"] <= GFA_MAX_SQFT]
excluded_gfa = before_gfa_cap_count - len(df_clean)
print(f"After PropertyGFATotal <= {GFA_MAX_SQFT:,} sq ft: {len(df_clean)} rows (excluded {excluded_gfa} row(s))")

final_count = len(df_clean)
print(f"\n>>> FINAL ROW COUNT: {final_count}")


print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS (key columns)")
print("=" * 60)
key_cols = [TARGET_COLUMN, "PropertyGFATotal", "YearBuilt", "NumberofFloors", "ENERGYSTARScore"]
print(df_clean[key_cols].describe().round(2))

# =============================================================================
# 7. MISSING VALUES SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("MISSING VALUES (after cleaning)")
print("=" * 60)
missing = df_clean.isnull().sum()
missing_pct = (missing / len(df_clean) * 100).round(1)
missing_df = pd.DataFrame({"missing": missing, "pct": missing_pct})
missing_df = missing_df[missing_df["missing"] > 0].sort_values("missing", ascending=False)

print(missing_df.head(15))


print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)
os.makedirs(FIGURES_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# 8a. Target distribution
fig, ax = plt.subplots(figsize=(6, 5))

ax.hist(df_clean[TARGET_COLUMN], bins=50, edgecolor="black", alpha=0.7)
ax.set_title(f"Distribution of {TARGET_COLUMN}")
ax.set_xlabel(TARGET_COLUMN)
ax.set_ylabel("Count")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig_target_distribution.png"), dpi=100, bbox_inches="tight")
print("Saved: fig_target_distribution.png")
plt.close()

# 8b. Target vs Building Type
fig, ax = plt.subplots(figsize=(12, 6))
top_types = df_clean["PrimaryPropertyType"].value_counts().head(10).index
df_top = df_clean[df_clean["PrimaryPropertyType"].isin(top_types)]
sns.boxplot(data=df_top, x="PrimaryPropertyType", y=TARGET_COLUMN)
plt.xticks(rotation=45, ha="right")
plt.title(f"{TARGET_COLUMN} by Primary Property Type (top 10)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig_target_vs_property_type.png"), dpi=100, bbox_inches="tight")
print("  Saved: fig_target_vs_property_type.png")
plt.close()

# 8c. Target vs YearBuilt
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df_clean["YearBuilt"], df_clean[TARGET_COLUMN], alpha=0.4, s=20)
ax.set_xlabel("Year Built")
ax.set_ylabel(TARGET_COLUMN)
ax.set_title(f"{TARGET_COLUMN} vs Year Built")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig_target_vs_year_built.png"), dpi=100, bbox_inches="tight")
print("  Saved: fig_target_vs_year_built.png")
plt.close()

# 8d. Target vs PropertyGFATotal (before excluding GFA > 8M — shows mega-site outlier)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df_before_gfa_cap["PropertyGFATotal"], df_before_gfa_cap[TARGET_COLUMN], alpha=0.4, s=20)
ax.set_xlabel("Property GFA Total (sq ft)")
ax.set_ylabel(TARGET_COLUMN)
ax.set_title(f"{TARGET_COLUMN} vs Building Size (before excluding GFA > {GFA_MAX_SQFT:,} sq ft)")
ax.ticklabel_format(style="plain", axis="x")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig_target_vs_gfa_before_cap.png"), dpi=100, bbox_inches="tight")
print("  Saved: fig_target_vs_gfa_before_cap.png")
plt.close()

# 8d-bis. Same scatter after removing GFA > 8M (used for analysis & export)
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(df_clean["PropertyGFATotal"], df_clean[TARGET_COLUMN], alpha=0.4, s=20)
ax.set_xlabel("Property GFA Total (sq ft)")
ax.set_ylabel(TARGET_COLUMN)
ax.set_title(f"{TARGET_COLUMN} vs Building Size (GFA <= {GFA_MAX_SQFT:,} sq ft)")
ax.ticklabel_format(style="plain", axis="x")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig_target_vs_gfa.png"), dpi=100, bbox_inches="tight")
print("  Saved: fig_target_vs_gfa.png")
plt.close()

# 8e. Correlation matrix
corr_cols = [TARGET_COLUMN, "PropertyGFATotal", "YearBuilt", "NumberofFloors", "PropertyGFABuilding(s)", "ENERGYSTARScore"]
corr_cols = [c for c in corr_cols if c in df_clean.columns]
corr_matrix = df_clean[corr_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
plt.title("Correlation Matrix (key numeric features)")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig_correlation_matrix.png"), dpi=100, bbox_inches="tight")
print("  Saved: fig_correlation_matrix.png")
plt.close()

# 8f. Building Type distribution
fig, ax = plt.subplots(figsize=(10, 6))
df_clean["BuildingType"].value_counts().plot(kind="bar", ax=ax)
ax.set_title("Number of Buildings by Building Type")
ax.set_xlabel("Building Type")
ax.set_ylabel("Count")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "fig_building_type_dist.png"), dpi=100, bbox_inches="tight")
print("  Saved: fig_building_type_dist.png")
plt.close()

# =============================================================================
# 9. ROW COUNT TRACKING
# =============================================================================
print("\n" + "=" * 60)
print("ROW COUNT TRACKING (as required)")
print("=" * 60)
tracking = [
    ("Initial load", initial_count),
    ("After non-residential filter", after_residential_filter),
    ("After all filters before GFA cap", before_gfa_cap_count),
    ("Final (after GFA <= 8M cap)", final_count),
]
for step, count in tracking:
    print(f"  {step}: {count} rows")


df_clean.to_csv(CLEAN_OUTPUT_PATH, index=False)
print(f"\nCleaned data saved to: {CLEAN_OUTPUT_PATH}")
building_consumption = df_clean
print("\n" + "=" * 60)
print("STEP 1 COMPLETE: Exploratory Analysis Done")
print("=" * 60)
