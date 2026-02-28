from pyspark.sql import SparkSession
from pyspark.sql.functions import col, mean, stddev, abs as spark_abs
from pyspark.sql.types import DoubleType

# Initialize Spark Session (already available as 'spark' in a Databricks notebook)
# spark = SparkSession.builder.appName("ZScoreOutlierRemoval").getOrCreate()

# 1. Load the housing data from a file
# Replace 'path/to/housing_data.csv' with the actual path in your Databricks environment (e.g., '/FileStore/tables/housing_data.csv')
df = spark.read.csv('path/to/housing_data.csv', header=True, inferSchema=True)

# Cache the DataFrame for performance as it will be used multiple times
df.cache()

print(f"Original DataFrame shape: {(df.count(), len(df.columns))}")

# 2. Identify all numeric columns
numeric_cols = [c for c, t in df.dtypes if t in ['int', 'double', 'float', 'long']]
print(f"Numeric columns identified: {numeric_cols}")

# 3. Calculate mean and standard deviation for each numeric column
# Collect the statistics into a dictionary for easier access
stats_df = df.selectExpr([f"mean({c}).alias('mean_{c}')" for c in numeric_cols] + 
                         [f"stddev({c}).alias('stddev_{c}')" for c in numeric_cols])
stats = stats_df.first().asDict()

# Define the Z-score threshold
Z_SCORE_THRESHOLD = 3.0

# 4. Filter out outliers using a loop to apply the Z-score condition to all numeric columns
# Start with the original DataFrame
df_cleaned = df

for column in numeric_cols:
    col_mean = stats[f'mean_{column}']
    col_stddev = stats[f'stddev_{column}']
    
    # Check if standard deviation is zero to avoid division by zero
    if col_stddev > 0:
        # Calculate the absolute Z-score for each row in the column
        z_score_col = spark_abs((col(column) - col_mean) / col_stddev)
        
        # Filter rows where the Z-score is within the threshold
        df_cleaned = df_cleaned.filter(z_score_col < Z_SCORE_THRESHOLD)
    else:
        print(f"Column {column} has zero standard deviation, skipping outlier removal for this column.")

# 5. Display the cleaned DataFrame information
print(f"Cleaned DataFrame shape (outliers removed): {(df_cleaned.count(), len(df_cleaned.columns))}")

# Show some cleaned data
# df_cleaned.show()
