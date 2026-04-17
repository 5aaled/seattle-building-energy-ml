

import numpy as np
import pandas as pd
import bentoml

from api.schemas import BuildingInput, NEIGHBORHOODS

MODEL_TAG = "seattle_energy_model:latest"
DATA_YEAR = 2016
KNOWN_NEIGHBORHOODS = {n.upper() for n in NEIGHBORHOODS if n != "Other"}

# Treat this class as the API service, and call it seattle_energy_api in BentoML’s world.” The actual HTTP routes come from methods on this class decorated with @bentoml.api (e.g. predict).
@bentoml.service(name="seattle_energy_api")
class SeattleEnergyService:

# __init__ pulls the saved Bento model, unpacks the preprocessor + metadata + default fills, and loads the fitted prediction pipeline, all onto self for the predict API method.
    def __init__(self):
        ref = bentoml.models.get(MODEL_TAG)
        co = ref.custom_objects
        self.preprocessor = co["preprocessor"]
        self.feature_names = co["feature_names"]
        self.default_values = co["default_values"]
        self.pipeline = ref.load_model()

# next step
# It transforms user input (BuildingInput) into a single row of features (a table row) that the ML understand
# Converts incoming API data into a structured DataFrame
# Performs:
# Feature engineering
# Data cleaning / normalization
# Derived feature creation (e.g., ratios, logs, flags)
# The output is a Pandas DataFrame with exactly 1 row.
    def _input_to_row(self, d: BuildingInput) -> pd.DataFrame:
        gfa = d.property_gfa_total
        largest = d.largest_property_use_type_gfa or gfa
        neighborhood_ok = d.neighborhood.upper() in KNOWN_NEIGHBORHOODS
        floors = max(d.number_of_floors, 1)
        # Step 2 uses PropertyGFABuilding(s) / NumberofFloors; we approximate with total GFA / floors
        area_per_floor = gfa / floors

        return pd.DataFrame(
            [
                {
                    "CouncilDistrictCode": d.council_district_code,
                    "Latitude": d.latitude,
                    "Longitude": d.longitude,
                    "NumberofBuildings": d.number_of_buildings,
                    "NumberofFloors": d.number_of_floors,
                    "LargestPropertyUseTypeGFA": largest,
                    "ENERGYSTARScore": d.energystar_score
                    if d.energystar_score is not None
                    else np.nan,
                    "BuildingAge": DATA_YEAR - d.year_built,
                    "ParkingRatio": min(1.0, d.property_gfa_parking / gfa) if gfa > 0 else 0.0,
                    "AreaPerFloor": area_per_floor,
                    "LogPropertyGFATotal": np.log1p(gfa),
                    "HasMultipleUses": d.has_multiple_uses,
                    "UseTypeCount": d.use_type_count,
                    "LargestUseProportion": min(1.0, largest / gfa) if gfa > 0 else 1.0,
                    "HasElectricity": d.has_electricity,
                    "HasNaturalGas": d.has_natural_gas,
                    "HasSteam": d.has_steam,
                    "EnergySourceCount": d.has_electricity + d.has_natural_gas + d.has_steam,
                    "HasENERGYSTARScore": 1 if d.energystar_score is not None else 0,
                    "BuildingType": d.building_type,
                    "NeighborhoodGrouped": d.neighborhood if neighborhood_ok else "Other",
                    "PrimaryPropertyTypeGrouped": d.primary_property_type,
                }
            ]
        )

    @bentoml.api
    # It takes validated input → transforms it → runs the model → returns a prediction.
    def predict(self, data: BuildingInput) -> dict:
        row = self._input_to_row(data)
        X = pd.DataFrame(self.preprocessor.transform(row), columns=self.feature_names)
        X = X.fillna(pd.Series(self.default_values)).fillna(0)
        pred = float(self.pipeline.predict(X)[0])
        return {"site_eui_kbtu_sf": round(pred, 2), "unit": "kBtu/sf"}


svc = SeattleEnergyService
