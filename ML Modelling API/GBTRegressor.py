# =====================================================
# Bike Rental Hourly Prediction (UCI Bike Sharing Dataset)
# Author: Hadi Fanaee-T (2013) - Capital Bikeshare System
# Environment: Databricks + PySpark
# Target: cnt (total bike rentals per hour)
# =====================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("BikeRental_Hourly_GBT_CV").getOrCreate()

# =============================================================
# 1. Data Preprocessing
# =============================================================

# Load directly from reliable GitHub raw mirror of UCI dataset (no zip/pandas needed)
url = "https://raw.githubusercontent.com/muditp19/UCI_Bike-sharing-dataset/master/hour.csv"
df = spark.read.csv(url, header=True, inferSchema=True)

print(f"Original row count: {df.count()}")
df.printSchema()

# Drop irrelevant columns as requested: instant, dteday (called dtedate in query), casual, registered
df = df.drop("instant", "dteday", "casual", "registered")

# Check & fill NA (UCI dataset has no missing values, but for completeness)
df = df.na.fill(0)   # or use mean/median if you prefer

# No scaling needed for Gradient Boosted Trees (they are scale-invariant)

# Feature columns
feature_cols = ["season", "yr", "mnth", "hr", "holiday", "weekday", 
                "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]

# Split into train / test with seed
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

print(f"Train rows: {train_df.count()}, Test rows: {test_df.count()}")

# =============================================================
# Visualize the data (Databricks display + simple aggregations)
# =============================================================

# 1. Overall summary
display(df.summary())

# 2. Average rentals by hour of day (very important pattern!)
display(
    df.groupBy("hr")
      .agg({"cnt": "mean"})
      .withColumnRenamed("avg(cnt)", "avg_rentals")
      .orderBy("hr")
)

# 3. Average rentals by workingday vs weekend
display(
    df.groupBy("workingday")
      .agg({"cnt": "mean"})
      .withColumnRenamed("avg(cnt)", "avg_rentals")
)

# 4. Sample of data
display(df.orderBy("cnt").limit(10))

# Optional: You can also do matplotlib/seaborn plots if desired
# pdf = df.toPandas()
# import matplotlib.pyplot as plt
# pdf.groupby('hr')['cnt'].mean().plot(kind='bar', figsize=(12,6))
# display(plt.gcf())

# =============================================================
# 2-6. Build ML Pipeline + CrossValidator (5-fold)
# =============================================================

# VectorAssembler
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="raw_features",
    handleInvalid="skip"
)

# VectorIndexer (automatically detects categorical features: season, yr, mnth, hr, etc.)
indexer = VectorIndexer(
    inputCol="raw_features",
    outputCol="features",
    maxCategories=25,   # hr has 24 values
    handleInvalid="skip"
)

# Define model - GBTRegressor
gbt = GBTRegressor(
    featuresCol="features",
    labelCol="cnt",
    seed=42
)

# Wrap everything in a Pipeline
pipeline = Pipeline(stages=[assembler, indexer, gbt])

# Evaluator: RMSE
evaluator = RegressionEvaluator(
    labelCol="cnt",
    predictionCol="prediction",
    metricName="rmse"
)

# ParamGrid for tuning (you can expand this)
paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [5, 8, 10])
             .addGrid(gbt.maxIter, [20, 50, 100])
             .addGrid(gbt.stepSize, [0.05, 0.1])
             .build())

# CrossValidator (5-fold as standard practice)
cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=5,          # ← 5-fold cross-validation
    seed=42,
    parallelism=4        # adjust based on your cluster size
)

# =============================================================
# 7. Train the model (CrossValidator does the training)
# =============================================================

print("Training GBT model with 5-fold CrossValidation...")
cv_model = cv.fit(train_df)

# Best model from CV
best_model = cv_model.bestModel

# Print best hyperparameters
best_gbt = best_model.stages[-1]
print(f"Best maxDepth   : {best_gbt.getMaxDepth()}")
print(f"Best maxIter    : {best_gbt.getMaxIter()}")
print(f"Best stepSize   : {best_gbt.getStepSize()}")

# Average RMSE across folds
print(f"Average CV RMSE : {min(cv_model.avgMetrics):.2f}")

# =============================================================
# 8. Make predictions and evaluate on Test set
# =============================================================

predictions = best_model.transform(test_df)

# Evaluate
rmse = evaluator.evaluate(predictions)
r2_evaluator = RegressionEvaluator(labelCol="cnt", predictionCol="prediction", metricName="r2")
r2 = r2_evaluator.evaluate(predictions)

print(f"\n=== Final Test Performance ===")
print(f"RMSE  : {rmse:.2f}")
print(f"R²    : {r2:.4f}")

# Show sample predictions
predictions.select("hr", "temp", "workingday", "cnt", "prediction").show(10, truncate=False)

# Optional: Save model
# best_model.write().overwrite().save("/dbfs/models/bike_rental_gbt_cv")