"""
Test the Seattle Energy Prediction API locally.

BentoML 2 wraps each API argument by parameter name: the body must be
{"data": { ... BuildingInput fields ... }} because predict(self, data: BuildingInput).
"""

import requests

API_URL = "http://localhost:3000/predict"

# Fields that match api/schemas.py BuildingInput
building = {
    "year_built": 1990,
    "property_gfa_total": 50000,
    "number_of_floors": 5,
    "primary_property_type": "Office",
    "neighborhood": "DOWNTOWN",
    "energystar_score": 65,
}

if __name__ == "__main__":
    print("POST", API_URL)
    try:
        resp = requests.post(API_URL, json={"data": building}, timeout=15)
        if not resp.ok:
            print(resp.status_code, resp.text)
            resp.raise_for_status()
        print("Response:", resp.json())
    except requests.exceptions.ConnectionError:
        print("Connection refused. Start the API: bentoml serve api.service:svc")
    except Exception as e:
        print("Error:", e)
