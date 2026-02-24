import pandas as pd
import numpy as np

# In Databricks, files in DBFS can be accessed using a local path syntax 
# (e.g., "/dbfs/FileStore/tables/housing.csv"). 
# Ensure you have uploaded your 'housing.csv' file to DBFS first.
csv_file_path = "/dbfs/FileStore/tables/housing.csv"

# --- 1. Load CSV file into a pandas DataFrame ---
try:
    # Use pandas read_csv to load the file from the DBFS path
    # Databricks supports standard pandas operations within the notebook
    df = pd.read_csv(csv_file_path)
    print("Successfully loaded the CSV file into a pandas DataFrame.")
except FileNotFoundError:
    print(f"Error: The file was not found at {csv_file_path}.")
    print("Please ensure the file is uploaded to the specified DBFS path.")
    # Create a dummy DataFrame for demonstration if file is missing
    data = {'ocean_proximity': ['NEAR BAY', 'NEAR OCEAN', np.nan, 'INLAND', 'NEAR BAY', np.nan],
            'total_bedrooms': [100.0, 120.0, np.nan, 150.0, np.nan, 110.0],
            'housing_median_age': [41, 21, 52, 41, 15, 30]}
    df = pd.DataFrame(data)

# Display the initial DataFrame with missing values
print("\nOriginal DataFrame Info:")
df.info()
print("\nMissing values before imputation:")
print(df.isna().sum())

# --- 2. Determine categorical columns using dtype ---
# Select columns that are of 'object' or 'category' dtype
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"\nCategorical columns identified: {categorical_cols}")

# --- 3. Impute missing values for a specific categorical column ---
target_categorical_column = 'ocean_proximity'

if target_categorical_column in categorical_cols:
    # Option A: Impute with a constant value like "Unknown"
    # df[target_categorical_column].fillna("Unknown", inplace=True)
    # print(f"\nImputed missing values in '{target_categorical_column}' with 'Unknown'.")

    # Option B: Impute with the most frequent value (mode)
    # The mode() method can be used, taking the first value from the result (mode returns a Series)
    most_frequent_value = df[target_categorical_column].mode()[0]
    df[target_categorical_column].fillna(most_frequent_value, inplace=True)
    print(f"\nImputed missing values in '{target_categorical_column}' with the most frequent value: '{most_frequent_value}'.")
else:
    print(f"\nColumn '{target_categorical_column}' not found or is not categorical.")

# Display the DataFrame information after imputation
print("\nDataFrame Info after imputation:")
df.info()
print("\nMissing values after imputation:")
print(df.isna().sum())
