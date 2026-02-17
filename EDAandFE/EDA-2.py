
# Import necessary library
import pandas as pd

# Define the path to your CSV file in Databricks File System (DBFS) or a Unity Catalog Volume.
# Example path for DBFS: "dbfs:/FileStore/tables/housing_data.csv"
# Example path for Unity Catalog Volume: "/Volumes/<catalog>/<schema>/<volume>/housing_data.csv"
file_path = "dbfs:/FileStore/tables/housing_data.csv" # Replace with your actual path

# 1. Load the housing data from the CSV file into a Spark DataFrame
spark_df = spark.read.csv(file_path, header=True, inferSchema=True)

# Optional: Display the original Spark DataFrame (specific to Databricks notebooks)
display(spark_df)

# 2. Convert the Spark DataFrame to a pandas DataFrame
# Note: This loads all data into the driver's memory, so it should only be used for small DataFrames.
pandas_df = spark_df.toPandas()

# Optional: Print the shape of the original pandas DataFrame
print(f"Original pandas DataFrame shape: {pandas_df.shape}")

# 3. Calculate the number of duplicate rows before dropping
# df.duplicated() returns a boolean Series (True for duplicates), .sum() counts the True values.
num_duplicates = pandas_df.duplicated().sum()
print(f"Number of duplicate rows found: {num_duplicates}")

# 4. Drop the duplicate rows from the pandas DataFrame (keeps the first occurrence by default)
pandas_df_unique = pandas_df.drop_duplicates()

# Optional: Print the shape of the DataFrame after dropping duplicates
print(f"Pandas DataFrame shape after dropping duplicates: {pandas_df_unique.shape}")

# 5. Print the number of duplicate rows that were removed
# This can be inferred by the difference in the number of rows (shape[0])
num_removed = pandas_df.shape[0] - pandas_df_unique.shape[0]
print(f"Number of duplicate rows removed: {num_removed}")

# Optional: Display the unique pandas DataFrame (specific to Databricks notebooks)
display(pandas_df_unique)
