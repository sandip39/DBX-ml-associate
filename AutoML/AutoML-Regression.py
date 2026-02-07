# Databricks notebook source
# COMMAND ----------

# 1. Import required libraries
from pyspark.sql import SparkSession
from databricks.automl import AutoMLRegressor
import databricks.automl

# COMMAND ----------

# 2. Read the housing data using Spark
# Replace with your actual file path
# Examples:
#   dbfs:/FileStore/housing.csv
#   /Volumes/my_catalog/my_schema/my_volume/housing.csv
#   s3://my-bucket/data/california_housing.csv

spark = SparkSession.builder.getOrCreate()

file_path = "/databricks-datasets/housing/housing.csv"  # example: built-in Databricks dataset

df = spark.read.option("header", "true") \
               .option("inferSchema", "true") \
               .csv(file_path)

# Show basic info
print("Dataset schema:")
df.printSchema()

print(f"\nNumber of rows: {df.count()}")

display(df.limit(10))

# COMMAND ----------

# 3. Prepare the data - random split into train and test
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print(f"Train rows: {train_df.count()}")
print(f"Test rows:  {test_df.count()}")

# Optional: cache the dataframes if you have enough memory
# train_df.cache()
# test_df.cache()

# COMMAND ----------

# 4. Run AutoML regression
# Target column must be numeric (e.g. 'median_house_value')

# Define the AutoML experiment
automl_reg = AutoMLRegressor(
    experiment_name="housing_price_automl_exp",          # will appear in Experiments
    experiment_dir="/Users/your.email@example.com/housing_automl",  # optional: custom path
    target_col="median_house_value",                     # ← your regression target
    primary_metric="rmse",                               # or "mae", "r2", "mapd"
    timeout_minutes=60,                                  # max runtime (adjust as needed)
    max_trials=30,                                       # max models to try
    max_concurrent_trials=4,                             # parallelism
    seed=42
)

# Start AutoML training
# You can pass either the full dataframe or just the training split
summary = automl_reg.fit(train_df)

# COMMAND ----------

# 5. View results and best model
print("AutoML run summary:")
display(summary)

# Best model information
print("\nBest model:")
print(f"Best trial: {summary.best_trial_id}")
print(f"Best model metric ({summary.primary_metric}): {summary.best_metric_value}")

# COMMAND ----------

# 6. Predict on test set using the best model
# Get the best model from AutoML
best_model = summary.best_model

# Make predictions
predictions = best_model.transform(test_df)

# Show sample predictions
print("Sample predictions on test set:")
display(predictions.select(
    "median_house_value",
    "prediction",
    "median_income",
    "housing_median_age",
    "total_rooms"
).limit(10))

# COMMAND ----------

# 7. (Optional) Evaluate manually on test set
from pyspark.ml.evaluation import RegressionEvaluator

evaluator_rmse = RegressionEvaluator(
    labelCol="median_house_value",
    predictionCol="prediction",
    metricName="rmse"
)

rmse = evaluator_rmse.evaluate(predictions)
print(f"Test RMSE: {rmse:.2f}")

# You can also compute MAE, R², etc.