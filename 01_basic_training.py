import mlflow, mlflow.sklearn
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# local experiments go to ./mlruns by default (no server required)
mlflow.set_experiment("nyc-taxi-basic")

print("Loading data…")
df = pd.read_csv("data/train.csv")

# basic features
df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["hour"] = df["pickup_datetime"].dt.hour
df["day_of_week"] = df["pickup_datetime"].dt.dayofweek
df["distance"] = np.sqrt(
    (df["dropoff_longitude"] - df["pickup_longitude"])**2 +
    (df["dropoff_latitude"] - df["pickup_latitude"])**2
)

features = [
    "passenger_count","hour","day_of_week","distance",
    "pickup_longitude","pickup_latitude","dropoff_longitude","dropoff_latitude"
]
X, y = df[features], df["trip_duration"]

# remove >5h trips
mask = y < 18000
X, y = X[mask], y[mask]

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("n_features", len(features))
    mlflow.log_param("n_training_samples", len(Xtr))

    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    model.fit(Xtr, ytr)

    yhat = model.predict(Xte)
    rmse = float(np.sqrt(mean_squared_error(yte, yhat)))
    mae  = float(mean_absolute_error(yte, yhat))
    r2   = float(r2_score(yte, yhat))

    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(model, "model")

print(f"Done! RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.3f}")
