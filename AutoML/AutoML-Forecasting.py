# Databricks notebook source
# COMMAND ----------

# 1. Imports
import pandas as pd
from databricks import automl

# COMMAND ----------

# 2. Read the data (assuming it's already in long format)
# Replace with your actual path / table / volume path
file_path = "/dbfs/FileStore/covid_long_format.csv"          # example CSV path
# or: spark.table("main.default.covid_long").toPandas()     # if it's in Unity Catalog

df = pd.read_csv(file_path)

# If reading from Unity Catalog / Delta table via Spark:
# df = spark.table("main.default.covid_confirmed_long").toPandas()

print("Loaded data shape:", df.shape)
print("Columns:", df.columns.tolist())
display(df.head(8))

# COMMAND ----------

# 3. Quick validation & standardization
# Make sure column names are consistent
# Adjust these if your column names are different
country_col     = "Country/Region"      # or "country", "Country_Region", etc.
time_col        = "date"
target_col      = "confirmed_cases"     # or "new_cases", "deaths", etc.

# Ensure date is datetime
if df[time_col].dtype != "datetime64[ns]":
    df[time_col] = pd.to_datetime(df[time_col])

# Sort for clean chronological order
df = df.sort_values([country_col, time_col]).reset_index(drop=True)

# Optional: remove or handle negative values / obvious errors
df[target_col] = df[target_col].clip(lower=0)

print(f"Unique countries/regions: {df[country_col].nunique()}")
print(f"Date range: {df[time_col].min().date()} → {df[time_col].max().date()}")

# Optional: filter to countries with enough data (e.g. at least 100 days)
min_days = 100
valid_countries = df.groupby(country_col).filter(lambda x: len(x) >= min_days)[country_col].unique()
df = df[df[country_col].isin(valid_countries)]

print(f"Countries after filtering: {df[country_col].nunique()}")

# COMMAND ----------

# 4. Chronological train/test split (across all series)
split_ratio = 0.8
min_date = df[time_col].min()
max_date = df[time_col].max()
split_date = min_date + (max_date - min_date) * split_ratio

print(f"Global split date ≈ {split_date.date()}")

train_pd = df[df[time_col] <= split_date].copy()
test_pd  = df[df[time_col] >  split_date].copy()

print(f"Train: {train_pd[time_col].min().date()} → {train_pd[time_col].max().date()}  ({len(train_pd):,} rows)")
print(f"Test:  {test_pd[time_col].min().date()}  → {test_pd[time_col].max().date()}   ({len(test_pd):,} rows)")
print(f"Countries in train/test: {train_pd[country_col].nunique()} / {test_pd[country_col].nunique()}")

# COMMAND ----------

# 5. Run AutoML Forecasting – multi-series (one series per country)
forecast_summary = automl.forecast(
    dataset              = train_pd,
    time_col             = time_col,
    target_col           = target_col,
    id_cols              = [country_col],               # ← per-country models
    horizon              = 30,                          # forecast horizon in days
    frequency            = "D",                         # daily frequency
    primary_metric       = "smape",                     # good for scale-varying series
    timeout_minutes      = 90,                          # adjust based on # of countries
    country_code         = "",                          # "" = no holiday effects; "US" = include US holidays
    experiment_name      = "covid_per_country_forecast_long_format",
    # imputers           = {target_col: "linear"},       # optional
    # exclude_frameworks = ["prophet", "deepar"]         # optional
)

# COMMAND ----------

# 6. View results
print("AutoML Forecasting Summary:")
display(forecast_summary)

print(f"\nBest trial: {forecast_summary.best_trial_id}")
print(f"Best {forecast_summary.primary_metric.upper()}: {forecast_summary.best_metric_value:.4f}")

# → Check Experiments UI for:
#   - per-country metrics
#   - forecast plots
#   - model comparison