import pandas as pd
from ydata_profiling import ProfileReport
from databricks.automl import ingest_data

# 1. Load the housing data into a Pandas DataFrame
# Replace the path with the actual location of your housing data file in Databricks
# The example uses a hypothetical path.
file_path = "/FileStore/tables/housing_data.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
    print("Please upload your data to Databricks and update the 'file_path' variable.")

# If you have a Spark DataFrame 'spark_df', convert it to a pandas DataFrame first
# df = spark_df.toPandas() 

# 2. Generate the ProfileReport with specific correlations enabled
# The 'correlations' parameter takes a dictionary to specify which correlations to calculate
profile = ProfileReport(
    df,
    title="Housing Data Profiling Report (with Correlations)",
    correlations={
        "pearson": {"calculate": True},
        "spearman": {"calculate": True},
        "kendall": {"calculate": True},
        "phi_k": {"calculate": False}, # Phi_K is for categorical/mixed data correlation
        "cramers": {"calculate": False} # Cramers V is for categorical correlation
    },
    # Set other options as needed, e.g., infer_dtypes=False for explicit type handling
    infer_dtypes=False,
    interactions=None,
    missing_diagrams=None
)

# 3. Display the report in the Databricks notebook
# Use displayHTML to render the report correctly in the notebook interface
displayHTML(profile.to_html())
