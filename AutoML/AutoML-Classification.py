# Databricks notebook source
# COMMAND ----------

# 1. Import required libraries
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col
from databricks.automl import AutoMLClassifier

# COMMAND ----------

# 2. Define schema for adult.data (no header in the file)
adult_schema = StructType([
    StructField("age", IntegerType(), True),
    StructField("workclass", StringType(), True),
    StructField("fnlwgt", IntegerType(), True),
    StructField("education", StringType(), True),
    StructField("education_num", IntegerType(), True),
    StructField("marital_status", StringType(), True),
    StructField("occupation", StringType(), True),
    StructField("relationship", StringType(), True),
    StructField("race", StringType(), True),
    StructField("sex", StringType(), True),
    StructField("capital_gain", IntegerType(), True),
    StructField("capital_loss", IntegerType(), True),
    StructField("hours_per_week", IntegerType(), True),
    StructField("native_country", StringType(), True),
    StructField("income", StringType(), True)  # target: <=50K or >50K
])

# COMMAND ----------

# 3. Read the Adult dataset using Spark
spark = SparkSession.builder.getOrCreate()

file_path = "/databricks-datasets/adult/adult.data"

df = spark.read \
    .schema(adult_schema) \
    .option("header", "false") \
    .option("inferSchema", "false") \
    .option("delimiter", ", ") \
    .csv(file_path)

# Clean up any leading/trailing spaces in string columns (common in this dataset)
string_cols = ["workclass", "education", "marital_status", "occupation", 
               "relationship", "race", "sex", "native_country", "income"]

for c in string_cols:
    df = df.withColumn(c, col(c).cast("string"))  # ensure string type
    # Optional: trim spaces
    # df = df.withColumn(c, trim(col(c)))

print("Dataset schema:")
df.printSchema()

print(f"\nNumber of rows: {df.count():,}")

display(df.limit(10))

# COMMAND ----------

# 4. Quick data prep (handle ? as null, drop fnlwgt if not useful, etc.)
df_clean = df.replace("?", None)  # treat ? as null

# Optional: drop rows with nulls (or impute later — AutoML can handle missing values)
# df_clean = df_clean.dropna()

print(f"Rows after basic cleaning: {df_clean.count():,}")

# Target is already "income" — binary classification: "<=50K" vs ">50K"

# COMMAND ----------

# 5. Random split into train and test
train_df, test_df = df_clean.randomSplit([0.8, 0.2], seed=42)

print(f"Train rows: {train_df.count():,}")
print(f"Test rows:  {test_df.count():,}")

# Optional: cache for speed
# train_df.cache()
# test_df.cache()

# COMMAND ----------

# 6. Run AutoML Classification
automl_clf = AutoMLClassifier(
    experiment_name          = "adult_income_automl_classification",
    experiment_dir           = "/Users/your.email@example.com/adult_automl",  # optional - change to your path
    target_col               = "income",                     # binary target: <=50K / >50K
    primary_metric           = "accuracy",                   # alternatives: "f1", "precision", "recall", "log_loss", "auc"
    timeout_minutes          = 60,                           # adjust based on your needs
    max_trials               = 30,
    max_concurrent_trials    = 4,
    seed                     = 42,
    # Optional exclusions (useful if you want to skip slow models)
    # exclude_frameworks       = ["xgboost", "lightgbm"],
    # exclude_algorithms       = ["decision_tree"],
    # feature_selection        = True,  # default is True
)

# Fit on training data
summary = automl_clf.fit(train_df)

# COMMAND ----------

# 7. View AutoML results
print("AutoML Classification Summary:")
display(summary)

print("\nBest model details:")
print(f"Best trial ID: {summary.best_trial_id}")
print(f"Best {summary.primary_metric}: {summary.best_metric_value:.4f}")

# COMMAND ----------

# 8. Predict on test set using the best model
best_model = summary.best_model

predictions = best_model.transform(test_df)

print("Sample predictions on test set:")
display(predictions.select(
    "income",
    "prediction",
    "probability",
    "age",
    "education",
    "occupation",
    "hours_per_week"
).limit(10))

# COMMAND ----------

# 9. Evaluate on test set
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator

# Accuracy
acc_evaluator = MulticlassClassificationEvaluator(
    labelCol="income",
    predictionCol="prediction",
    metricName="accuracy"
)
accuracy = acc_evaluator.evaluate(predictions)
print(f"Test Accuracy: {accuracy:.4f}")

# F1 (weighted)
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="income",
    predictionCol="prediction",
    metricName="f1"
)
f1 = f1_evaluator.evaluate(predictions)
print(f"Test Weighted F1: {f1:.4f}")

# AUC (for binary classification)
auc_evaluator = BinaryClassificationEvaluator(
    labelCol="income_indexed",  # AutoML often adds an indexed label column
    rawPredictionCol="probability",
    metricName="areaUnderROC"
)
# If "income_indexed" doesn't exist, use this workaround:
# predictions = predictions.withColumn("income_indexed", when(col("income") === ">50K", 1.0).otherwise(0.0))
auc = auc_evaluator.evaluate(predictions)
print(f"Test AUC: {auc:.4f}")

# Confusion matrix
print("Confusion Matrix:")
predictions.groupBy("income", "prediction").count().show()