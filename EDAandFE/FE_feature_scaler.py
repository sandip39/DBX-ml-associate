# =============================================
# FEATURE SCALING – ALL NUMERICAL COLUMNS
# California Housing dataset (PySpark in Databricks)
# =============================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StandardScaler, MinMaxScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.appName("HousingFeatureScaling").getOrCreate()

# ── 1. Load data from public CSV ──────────────────────────────────────────
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

df = (spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(url))

df.cache()
print("Original shape (rows, columns):", (df.count(), len(df.columns)))
df.printSchema()

# Automatically detect numerical columns
numerical_cols = [field.name for field in df.schema.fields 
                  if field.dataType.simpleString() in ('int', 'double', 'float', 'long')]

print("Numerical columns:", numerical_cols)

# Handle missing values (only total_bedrooms has some)
df = df.na.fill({"total_bedrooms": df.approxQuantile("total_bedrooms", [0.5], 0.0)[0]})

# For scaling we usually exclude the target (median_house_value)
feature_cols = [c for c in numerical_cols if c != "median_house_value"]

print("Features to scale:", feature_cols)

# Assemble all numerical features into a single dense vector column
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="num_features_raw",
    handleInvalid="skip"   # in case any weird values
)

df_assembled = assembler.transform(df)

df_assembled.select("num_features_raw").show(5, truncate=False)

# StandardScaler: mean=0, variance=1 (most common choice)
standard_scaler = StandardScaler(
    inputCol="num_features_raw",
    outputCol="num_features_scaled_std",
    withMean=True,    # center to mean=0
    withStd=True      # scale to unit variance
)

# Fit & transform
scaler_std_model = standard_scaler.fit(df_assembled)
df_scaled_std = scaler_std_model.transform(df_assembled)

# Show comparison (first 5 rows)
print("StandardScaler results (first 5 rows):")
df_scaled_std.select(
    "total_rooms", "median_income",               # original example columns
    "num_features_raw", 
    "num_features_scaled_std"
).show(5, truncate=False)

# Optional: convert scaled vector back to columns if needed (for inspection)
from pyspark.ml.functions import vector_to_array

df_scaled_std.withColumn("scaled_array", vector_to_array("num_features_scaled_std")) \
    .select("scaled_array").show(3, truncate=False)


# MinMaxScaler: scales each feature to range [0, 1]
minmax_scaler = MinMaxScaler(
    inputCol="num_features_raw",
    outputCol="num_features_scaled_minmax"
)

# Fit & transform
scaler_minmax_model = minmax_scaler.fit(df_assembled)
df_scaled_minmax = scaler_minmax_model.transform(df_assembled)

print("MinMaxScaler results (first 5 rows):")
df_scaled_minmax.select(
    "total_rooms", "median_income",
    "num_features_raw",
    "num_features_scaled_minmax"
).show(5, truncate=False)


# Full pipeline: assemble → scale (choose one scaler)
pipeline_std = Pipeline(stages=[
    assembler,
    standard_scaler
])

# Fit on whole data (or better: fit only on train in real ML workflow)
pipeline_model_std = pipeline_std.fit(df)

# Transform
df_final_std = pipeline_model_std.transform(df)

# Now df_final_std has "num_features_scaled_std" ready for ML
# Example: you would next add categorical encoding, then VectorAssembler again for full feature vector

df_final_std.select(
    "median_house_value",           # target
    "num_features_scaled_std"       # scaled features
).show(3, truncate=False)



#Modify above for Pandas

# =============================================
# FEATURE SCALING – ALL NUMERICAL COLUMNS (PANDAS VERSION)
# California Housing dataset – Databricks notebook friendly
# =============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# ── 1. Load data directly into pandas ─────────────────────────────────────
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

df = pd.read_csv(url)

print("DataFrame shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isna().sum())
df.head()

# Automatically detect numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print("All numerical columns:", numerical_cols)

# Exclude the target variable from scaling
target = 'median_house_value'
feature_cols = [col for col in numerical_cols if col != target]

print("\nFeatures to scale:", feature_cols)

# Handle missing values (only total_bedrooms has NaNs)
# Simple median imputation – common & safe for this dataset
median_bedrooms = df['total_bedrooms'].median()
df['total_bedrooms'] = df['total_bedrooms'].fillna(median_bedrooms)

print("\nMissing values after imputation:", df[feature_cols].isna().sum().sum())

# StandardScaler – most common choice for many ML algorithms
std_scaler = StandardScaler()

# Fit & transform only the numerical feature columns
scaled_std = std_scaler.fit_transform(df[feature_cols])

# Create new DataFrame with scaled values (keeping column names)
df_scaled_std = pd.DataFrame(
    scaled_std,
    columns=[f"{col}_std" for col in feature_cols],
    index=df.index
)

# Combine original + scaled columns (or replace originals if preferred)
df_with_std = pd.concat([df, df_scaled_std], axis=1)

# Preview
print("\nStandardScaler – first 5 rows (original vs scaled):")
cols_to_show = ['total_rooms', 'median_income', 'total_rooms_std', 'median_income_std']
df_with_std[cols_to_show].head()

# MinMaxScaler – useful when you need bounded features
minmax_scaler = MinMaxScaler(feature_range=(0, 1))

# Fit & transform
scaled_minmax = minmax_scaler.fit_transform(df[feature_cols])

# New DataFrame with scaled values
df_scaled_minmax = pd.DataFrame(
    scaled_minmax,
    columns=[f"{col}_minmax" for col in feature_cols],
    index=df.index
)

# Combine
df_with_minmax = pd.concat([df, df_scaled_minmax], axis=1)

print("\nMinMaxScaler – first 5 rows (original vs scaled):")
cols_to_show_minmax = ['total_rooms', 'median_income', 'total_rooms_minmax', 'median_income_minmax']
df_with_minmax[cols_to_show_minmax].head()


