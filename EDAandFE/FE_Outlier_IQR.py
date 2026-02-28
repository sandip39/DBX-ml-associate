import pandas as pd
import numpy as np

def remove_iqr_outliers(df, k=1.5):
    """
    Remove outliers from all numeric columns of a pandas DataFrame using the IQR method.

    Args:
        df (pd.DataFrame): The input DataFrame.
        k (float): The multiplier for the IQR to define the bounds (default is 1.5).

    Returns:
        pd.DataFrame: The DataFrame with outliers removed.
    """
    df_cleaned = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        # Calculate Q1 (25th percentile) and Q3 (75th percentile)
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        
        # Calculate IQR
        IQR = Q3 - Q1
        
        # Define lower and upper bounds
        lower_bound = Q1 - k * IQR
        upper_bound = Q3 + k * IQR
        
        # Filter out outliers for the current column
        # Keep only values within the defined bounds
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
        
    return df_cleaned

# --- Databricks Usage Example ---

# 1. Define the path to your data file in DBFS (Databricks File System).
#    Replace 'path/to/your/housing_data.csv' with the actual path in your workspace.
#    Paths in DBFS typically start with '/dbfs/'.
file_path = "/dbfs/FileStore/tables/housing_data.csv" 

# 2. Read the data file into a pandas DataFrame
try:
    # Use the appropriate pandas read function based on your file type (e.g., read_csv, read_excel)
    housing_df = pd.read_csv(file_path)
    print(f"Original DataFrame shape: {housing_df.shape}")
    print("\nOriginal DataFrame Info:")
    housing_df.info()

    # 3. Apply the outlier removal function
    housing_df_cleaned = remove_iqr_outliers(housing_df)

    print(f"\nCleaned DataFrame shape (outliers removed): {housing_df_cleaned.shape}")
    print("\nCleaned DataFrame Info:")
    housing_df_cleaned.info()

except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("Please ensure the file is uploaded to the correct DBFS path.")
except Exception as e:
    print(f"An error occurred: {e}")

