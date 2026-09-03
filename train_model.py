import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
from pathlib import Path

# Create models directory if it doesn't exist
models_dir = Path(__file__).parent / "models"
models_dir.mkdir(exist_ok=True)

print("Generating synthetic solar dataset...")
np.random.seed(42)
n_samples = 5000

# Synthetic feature generation
irradiance = np.random.uniform(0, 1100, n_samples)
temperature = np.random.uniform(10, 45, n_samples)
wind_speed = np.random.uniform(0, 15, n_samples)
day_of_year = np.random.randint(1, 366, n_samples)
hour = np.random.randint(0, 24, n_samples)

# Sun height approximation
sun_height = np.where((hour >= 6) & (hour <= 18), 90 * np.sin(np.pi * (hour - 6) / 12), 0)

# Month approximation
month = np.random.randint(1, 13, n_samples)

# Target: Power Output in Watts (W)
# Physics-inspired synthetic target formula
power = irradiance * 3.2 * (1 - (temperature - 25) * 0.003) + np.random.normal(0, 25, n_samples)
power = np.where((sun_height <= 0) | (irradiance <= 0), 0, np.maximum(0, power))

df = pd.DataFrame({
    "irradiance": irradiance,
    "temperature": temperature,
    "wind_speed": wind_speed,
    "day_of_year": day_of_year,
    "hour": hour,
    "sun_height": sun_height,
    "month": month,
})

X = df
y = power

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Regressor...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = round(float(mean_absolute_error(y_test, y_pred)), 2)
rmse = round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 2)
r2 = round(float(r2_score(y_test, y_pred)), 4)

metrics = {
    "mae": mae,
    "rmse": rmse,
    "r2": r2
}

model_path = models_dir / "solar_power_model.pkl"
metrics_path = models_dir / "metrics.json"

joblib.dump(model, model_path)
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Model saved successfully to {model_path}")
print(f"Metrics saved to {metrics_path}: MAE={mae}, RMSE={rmse}, R2={r2}")
