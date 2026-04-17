"""
Send invalid payloads to the Seattle Energy API to exercise validation / errors.

Start the server first: bentoml serve api.service:svc
"""

import json

import requests

API_URL = "http://localhost:3000/predict"

# Each case: short label + body (BentoML 2 expects {"data": {...}} for predict(self, data: ...))

INVALID_CASES = [
    (
        "year_built too old (schema: ge=1800)",
        {
            "data": {
                "year_built": 1750,
                "property_gfa_total": 10000,
                "number_of_floors": 3,
                "primary_property_type": "Office",
                "neighborhood": "DOWNTOWN",
            }
        },
    ),
    (
        "year_built after DATA_YEAR (schema: le=2016)",
        {
            "data": {
                "year_built": 2020,
                "property_gfa_total": 10000,
                "number_of_floors": 3,
                "primary_property_type": "Office",
                "neighborhood": "DOWNTOWN",
            }
        },
    ),
    (
        "property_gfa_total zero (schema: gt=0)",
        {
            "data": {
                "year_built": 2000,
                "property_gfa_total": 0,
                "number_of_floors": 3,
                "primary_property_type": "Office",
                "neighborhood": "DOWNTOWN",
            }
        },
    ),
    (
        "number_of_floors zero (schema: ge=1)",
        {
            "data": {
                "year_built": 2000,
                "property_gfa_total": 5000,
                "number_of_floors": 0,
                "primary_property_type": "Office",
                "neighborhood": "DOWNTOWN",
            }
        },
    ),
    (
        "primary_property_type not allowed (Literal)",
        {
            "data": {
                "year_built": 2000,
                "property_gfa_total": 5000,
                "number_of_floors": 2,
                "primary_property_type": "Castle",
                "neighborhood": "DOWNTOWN",
            }
        },
    ),
    (
        "energystar_score out of range (schema: 1–100)",
        {
            "data": {
                "year_built": 2000,
                "property_gfa_total": 5000,
                "number_of_floors": 2,
                "primary_property_type": "Retail",
                "neighborhood": "EAST",
                "energystar_score": 150,
            }
        },
    ),
    (
        "neighborhood empty (schema: min_length=1)",
        {
            "data": {
                "year_built": 2000,
                "property_gfa_total": 5000,
                "number_of_floors": 2,
                "primary_property_type": "Office",
                "neighborhood": "",
            }
        },
    ),
    (
        "missing required field (no neighborhood)",
        {
            "data": {
                "year_built": 2000,
                "property_gfa_total": 5000,
                "number_of_floors": 2,
                "primary_property_type": "Office",
            }
        },
    ),
    (
        "wrong top-level key (expects 'data', not 'building')",
        {
            "building": {
                "year_built": 1990,
                "property_gfa_total": 50000,
                "number_of_floors": 5,
                "primary_property_type": "Office",
                "neighborhood": "DOWNTOWN",
            }
        },
    ),
]


def main() -> None:
    print("POST", API_URL)
    for label, body in INVALID_CASES:
        print("\n---", label, "---")
        try:
            resp = requests.post(API_URL, json=body, timeout=15)
            print("Status:", resp.status_code)
            try:
                parsed = resp.json()
                print("Body:", json.dumps(parsed, indent=2)[:2000])
            except ValueError:
                print("Body (raw):", resp.text[:2000])
        except requests.exceptions.ConnectionError:
            print("Connection refused. Start the API: bentoml serve api.service:svc")
            return
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()
