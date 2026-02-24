import pandas as pd
import numpy as np

# 1. Load Data in Databricks (assuming CSV)
# Replace 'dbfs:/path/to/housing_data.csv' with your actual file path
df_spark = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("dbfs:/path/to/housing_data.csv")

# 2. Convert to Pandas DataFrame
df = df_spark.toPandas()

# Display missing values before imputation
print("Missing values before:\n", df.isnull().sum())

# 3. Missing Value Imputation
# A. Fill with Zero
df_filled_zero = df.fillna(0)

# B. Fill with Mean (applied to numeric columns)
df_filled_mean = df_filled_zero.fillna(df_filled_zero.mean(numeric_only=True))

# C. Fill with Median (applied to numerical columns)
# This acts as the final check, filling remaining NaNs with median
final_df = df_filled_mean.fillna(df_filled_mean.median(numeric_only=True))

# Display missing values after imputation
print("\nMissing values after:\n", final_df.isnull().sum())

# Display the imputed DataFrame
display(final_df)
