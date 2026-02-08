# Databricks notebook source
# COMMAND ----------

# ────────────────────────────────────────────────────────────────
# End-to-End MLflow Example: Wine Quality Prediction (Red + White)
# ────────────────────────────────────────────────────────────────

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os

# ──── 1. Load & Combine Datasets ─────────────────────────────────

# Typical paths in Databricks (adjust if you uploaded to different location)
red_path  = "/FileStore/winequality-red.csv"
white_path = "/FileStore/winequality-white.csv"

# If files are not there yet → you can download them manually or use:
# dbutils.fs.cp("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", red_path)

df_red = pd.read_csv(red_path, sep=";")
df_white = pd.read_csv(white_path, sep=";")

# Add wine type
df_red["wine_type"]  = "red"
df_white["wine_type"] = "white"

# Combine
df = pd.concat([df_red, df_white], ignore_index=True)

# Clean column names (remove spaces)
df.columns = [c.replace(" ", "_") for c in df.columns]

print(f"Shape: {df.shape}")
print(df["wine_type"].value_counts())
display(df.head())

# ──── 2. Basic Feature Engineering ───────────────────────────────

df["alcohol_level"] = pd.cut(
    df["alcohol"],
    bins=[0, 10, 12, np.inf],
    labels=["low", "medium", "high"]
)

df["total_sulfur_ratio"] = df["free_sulfur_dioxide"] / (df["total_sulfur_dioxide"] + 1e-6)

# ──── 3. Prepare data ────────────────────────────────────────────

X = df.drop("quality", axis=1)
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=X["wine_type"]
)

# ──── 4. Define Preprocessing Pipeline ───────────────────────────

numeric_features = [
    'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
    'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density',
    'pH', 'sulphates', 'alcohol', 'total_sulfur_ratio'
]

categorical_features = ['wine_type', 'alcohol_level']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ])

# ──── 5. MLflow Experiment ───────────────────────────────────────

mlflow.set_experiment("/Users/" + dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get() + "/wine_quality")

# ──── 6. Train & Log Multiple Models ─────────────────────────────

models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

best_rmse = float("inf")
best_run_id = None
best_model_name = None

for name, model in models.items():
    
    with mlflow.start_run(run_name=name) as run:
        
        # Full pipeline
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model)
        ])
        
        # Train
        pipe.fit(X_train, y_train)
        
        # Predict
        y_pred = pipe.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2   = r2_score(y_test, y_pred)
        
        # Log
        mlflow.log_param("model_type", name)
        mlflow.log_param("n_estimators", model.n_estimators if hasattr(model, "n_estimators") else None)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name="wine_quality_model" if rmse < best_rmse else None
        )
        
        # Log sample input for signature
        sample_input = X_test.head(5)
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=sample_input
        )
        
        print(f"{name} → RMSE: {rmse:.4f} | R²: {r2:.4f}")
        
        # Track best
        if rmse < best_rmse:
            best_rmse = rmse
            best_run_id = run.info.run_id
            best_model_name = name

print(f"\nBest model: {best_model_name} (RMSE = {best_rmse:.4f})")

# ──── 7. Register Best Model (if not already done) ───────────────

if best_run_id:
    result = mlflow.register_model(
        model_uri=f"runs:/{best_run_id}/model",
        name="wine_quality_model"
    )
    print(f"Registered model version: {result.version}")

# ──── 8. Load & Predict with Registered Model ────────────────────

# Get latest version
from mlflow.tracking import MlflowClient
client = MlflowClient()
latest_version = client.get_latest_versions("wine_quality_model", stages=["None", "Staging", "Production"])[0].version

print(f"Loading version {latest_version}...")

loaded_model = mlflow.pyfunc.load_model(f"models:/wine_quality_model/{latest_version}")

# Predict on a few test examples
sample_pred = loaded_model.predict(X_test.head(10))
print("\nSample predictions (first 10 test rows):")
print(pd.DataFrame({
    "actual": y_test.head(10).values,
    "predicted": sample_pred
}))

# ──── 9. (Optional) Promote to Production ────────────────────────

# client.transition_model_version_stage(
#     name="wine_quality_model",
#     version=latest_version,
#     stage="Production"
# )