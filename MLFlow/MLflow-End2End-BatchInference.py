import mlflow
from pyspark.sql.functions import struct

# --- PART 1: LOAD RAW DATA INTO DATABRICKS TABLE ---
# Define source file and target table paths
file_path = "/databricks-datasets/wine-quality/winequality-red.csv" 
input_table_name = "main.default.raw_wine_data"

# Read CSV with proper options (adjust 'sep' based on your file)
df_raw = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("sep", ";") \
    .load(file_path)

# Save as a permanent Delta table for governance and history
df_raw.write.mode("overwrite").format("delta").saveAsTable(input_table_name)

# --- PART 2: PERFORM BATCH INFERENCE ---
model_name = "wine_quality_xgboost"
model_uri = f"models:/{model_name}/Production"

# Load model as a Spark UDF for distributed scoring
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="double")

# Load the data we just saved
df_to_score = spark.table(input_table_name)

# Apply model (ensuring we exclude any non-feature columns if they exist)
# Here, we assume the table contains only features used during training
df_predictions = df_to_score.withColumn(
    "prediction", 
    predict_udf(struct(*df_to_score.columns))
)

# Save predictions to a final output table
output_table = "main.default.wine_predictions"
df_predictions.write.mode("append").saveAsTable(output_table)
