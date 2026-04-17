
import os
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(PROJECT_ROOT, "data", "clean", "building_consumption_clean.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "data", "enriched", "building_consumption_enriched.csv")

TARGET_COLUMN = "SiteEUI(kBtu/sf)"
DATA_YEAR = 2016

print("=" * 60)
print("STEP 2: FEATURE ENGINEERING")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)
print(f"Loaded {len(df)} rows from {INPUT_PATH}")
print(f"Initial columns: {len(df.columns)}")

print("\n--- Temporality features ---")
df["BuildingAge"] = DATA_YEAR - df["YearBuilt"]
df["BuildingAgeSquared"] = df["BuildingAge"] ** 2


# =============================================================================
# 3. STRUCTURE FEATURES
# =============================================================================
print("\n--- Structure features ---")
df["ParkingRatio"] = df["PropertyGFAParking"] / df["PropertyGFATotal"]
df["ParkingRatio"] = df["ParkingRatio"].fillna(0).clip(upper=1)
df["BuildingRatio"] = df["PropertyGFABuilding(s)"] / df["PropertyGFATotal"]
df["BuildingRatio"] = df["BuildingRatio"].fillna(1).clip(upper=1)
df["AreaPerFloor"] = df["PropertyGFABuilding(s)"] / df["NumberofFloors"].replace(0, np.nan)
df["AreaPerFloor"] = df["AreaPerFloor"].fillna(df["PropertyGFABuilding(s)"])
df["LogPropertyGFATotal"] = np.log1p(df["PropertyGFATotal"])



# 4. MULTI-USE FEATURES
print("\n--- Multi-use features ---")
df["HasMultipleUses"] = df["ListOfAllPropertyUseTypes"].fillna("").str.contains(",", regex=False).astype(int)
df["UseTypeCount"] = (df["ListOfAllPropertyUseTypes"].fillna("").str.count(",") + 1).clip(upper=5)
df["LargestUseProportion"] = df["LargestPropertyUseTypeGFA"] / df["PropertyGFATotal"]
df["LargestUseProportion"] = df["LargestUseProportion"].fillna(0).clip(upper=1)


# 5. ENERGY SOURCE INDICATORS

print("\n--- Energy source indicators ---")
df["HasElectricity"] = (df["Electricity(kWh)"].fillna(0) > 0).astype(int)
df["HasNaturalGas"] = (df["NaturalGas(kBtu)"].fillna(0) > 0).astype(int)
df["HasSteam"] = (df["SteamUse(kBtu)"].fillna(0) > 0).astype(int)
df["EnergySourceCount"] = df["HasElectricity"] + df["HasNaturalGas"] + df["HasSteam"]



# 6. LOCATION FEATURES

print("\n--- Location features ---")
neighborhood_counts = df["Neighborhood"].value_counts()
rare_neighborhoods = neighborhood_counts[neighborhood_counts < 30].index
df["NeighborhoodGrouped"] = df["Neighborhood"].replace(rare_neighborhoods, "Other")



print("\n--- PrimaryPropertyType grouping ---")
OFFICE_TYPES = ["Small- and Mid-Sized Office", "Large Office"]
RETAIL_TYPES = ["Retail Store", "Supermarket / Grocery Store"]
INDUSTRIAL_TYPES = ["Warehouse", "Distribution Center"]
INSTITUTIONAL_TYPES = ["K-12 School", "College/University", "Worship Facility", "Senior Care Community"]
HOSPITALITY_TYPES = ["Hotel"]


def map_property_type(pt):
    if pd.isna(pt):
        return "Other"
    if pt in OFFICE_TYPES:
        return "Office"
    if pt in RETAIL_TYPES:
        return "Retail"
    if pt in INDUSTRIAL_TYPES:
        return "Industrial"
    if pt in INSTITUTIONAL_TYPES:
        return "Institutional"
    if pt in HOSPITALITY_TYPES:
        return "Hospitality"
    return "Other"


df["PrimaryPropertyTypeGrouped"] = df["PrimaryPropertyType"].apply(map_property_type)



print("\n--- ENERGY STAR feature ---")
df["HasENERGYSTARScore"] = df["ENERGYSTARScore"].notna().astype(int)

# 9. SAVE ENRICHED DATASET

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"\nEnriched data saved to: {OUTPUT_PATH}")
print(f"Total columns now: {len(df.columns)}")
building_consumption_enriched = df
print("\n" + "=" * 60)
print("STEP 2 COMPLETE: Feature Engineering Done")
print("=" * 60)
