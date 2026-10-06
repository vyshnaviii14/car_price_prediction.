"""
Training script — runs the full pipeline and saves model + metrics.
Equivalent to running all notebook cells.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ── Load & clean ───────────────────────────────────────────────────────────────
DATA_PATH = 'CAR DETAILS FROM CAR DEKHO.csv'
df = pd.read_csv(DATA_PATH)
print(f"Loaded: {df.shape}")

n_dupes = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)
print(f"Duplicates removed: {n_dupes}  |  Clean shape: {df.shape}")

# ── Feature engineering ───────────────────────────────────────────────────────
df['brand']   = df['name'].str.split().str[0]
df['car_age'] = 2024 - df['year']
df['log_selling_price'] = np.log1p(df['selling_price'])

FEATURES = ['brand', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'car_age']
TARGET   = 'log_selling_price'

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ── Train/test split ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print(f"Train: {X_train.shape[0]}  Test: {X_test.shape[0]}")

# ── Preprocessor ──────────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = ['brand', 'fuel', 'seller_type', 'transmission', 'owner']
NUMERIC_FEATURES     = ['km_driven', 'car_age']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), NUMERIC_FEATURES),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
])

# ── Train models ──────────────────────────────────────────────────────────────
models = {
    'Linear Regression'  : LinearRegression(),
    'Random Forest'      : RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting'  : GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
}

results = {}
trained_pipelines = {}

for model_name, model in models.items():
    print(f"\nTraining: {model_name} ...")
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    pipeline.fit(X_train, y_train)

    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_true = np.expm1(y_test)

    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)

    results[model_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
    trained_pipelines[model_name] = pipeline
    print(f"  MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.4f}")

# ── Select best model ─────────────────────────────────────────────────────────
best_name = max(results, key=lambda m: results[m]['R2'])
print(f"\nBest Model: {best_name}")
print(f"  MAE  = {results[best_name]['MAE']:,.0f} INR")
print(f"  RMSE = {results[best_name]['RMSE']:,.0f} INR")
print(f"  R2   = {results[best_name]['R2']:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs('models', exist_ok=True)
joblib.dump(trained_pipelines[best_name], 'models/best_model.pkl')
print("\nModel saved to models/best_model.pkl")

# Save metrics to JSON for the report
with open('models/metrics.json', 'w') as f:
    json.dump({'results': results, 'best_model': best_name}, f, indent=2)
print("Metrics saved to models/metrics.json")

# ── Quick verification ────────────────────────────────────────────────────────
loaded = joblib.load('models/best_model.pkl')
sample = pd.DataFrame([{
    'brand': 'Maruti', 'km_driven': 50000, 'fuel': 'Petrol',
    'seller_type': 'Individual', 'transmission': 'Manual',
    'owner': 'First Owner', 'car_age': 7
}])
pred = float(np.expm1(loaded.predict(sample)[0]))
print(f"\nVerification prediction: INR {pred:,.0f}")
print("\nDONE.")
