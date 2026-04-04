# Seattle Building Energy Benchmarking - ML Project

Predict energy consumption for non-residential buildings using structural data.

## Project Structure

```
liverable/
├── data/
│   ├── raw/          # Original dataset (2016_Building_Energy_Benchmarking.csv)
│   ├── clean/        # Output of Step 1 (building_consumption_clean.csv)
│   ├── enriched/     # Output of Step 2 (building_consumption_enriched.csv)
│   ├── prepared/     # Output of Step 3 (X_prepared.csv, y.csv)
│   └── figures/      # Visualizations
├── src/
│   ├── step1_exploratory.py         # Step 1: Exploratory Analysis
│   ├── step2_feature_engineering.py # Step 2: Feature Engineering
│   ├── step3_prepare_features.py   # Step 3: Preparation for modeling
│   ├── step4_compare_models.py     # Step 4: Model comparison
│   └── step5_optimize_model.py    # Step 5: GridSearchCV + feature importance
├── api/                   # Mission Part 2 - API
│   ├── service.py
│   └── schemas.py
├── save_model.py          # Save model with BentoML
├── bentofile.yaml         # BentoML build config
├── scripts/
│   └── test_predict.py    # Test API
├── requirements.txt
└── README.md
```

## Run Steps

```bash
# Step 1: Exploratory Analysis
python src/step1_exploratory.py

# Step 2: Feature Engineering
python src/step2_feature_engineering.py

# Step 3: Preparation of features for modeling
python src/step3_prepare_features.py

# Step 4: Comparison of supervised models
python src/step4_compare_models.py

# Step 5: Optimization and interpretation (GridSearchCV + feature importance)
python src/step5_optimize_model.py

# Mission Part 2: Save model + serve API (after Step 5)
python save_model.py
bentoml serve api.service:svc
# In another terminal: python scripts/test_predict.py
```

## Data Flow

1. **Raw** → `data/raw/2016_Building_Energy_Benchmarking.csv`
2. **Step 1** → `data/clean/building_consumption_clean.csv` + `data/figures/*.png`
3. **Step 2** → `data/enriched/building_consumption_enriched.csv`
4. **Step 3** → `data/prepared/X_prepared.csv`, `y.csv` + `data/figures/step3_*.png`
5. **Step 4** → `data/prepared/model_comparison_results.csv`
6. **Step 5** → `data/figures/step5_feature_importance.png`, `data/prepared/step5_optimization_results.csv`, `data/prepared/step5_feature_importance.csv`
7. **Mission 2** → BentoML model store (`seattle_energy_model` tag) after `python save_model.py`

## Mission Part 2 - API with BentoML

```
liverable/
├── api/
│   ├── __init__.py
│   ├── service.py      # BentoML service + predict endpoint
│   └── schemas.py      # Pydantic validation
├── save_model.py       # Save trained model with BentoML
├── bentofile.yaml      # BentoML build config
└── scripts/
    └── test_predict.py # Test API with requests
```

