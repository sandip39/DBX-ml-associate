# =====================================================
# SFO Customer Satisfaction Survey Analysis & Prediction
# Dataset: 2017 SFO Customer Survey (from data.sfgov.org)
# Environment: Databricks + PySpark
# Target: Predict Q13Overall (overall satisfaction rating)
# Features: Q7A_ART to Q7O_WIFISVC (aspect ratings)
# =====================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, count, when, lit, round
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("SFO_Survey_DT_Regressor").getOrCreate()

# =============================================================
# 1. Load the dataset
# =============================================================
# Direct CSV URL from SF Open Data portal
csv_url = "https://data.sfgov.org/api/views/gcjv-3mzf/rows.csv?accessType=DOWNLOAD"
df = spark.read.csv(csv_url, header=True, inferSchema=True)

# =============================================================
# 2. Understand the data - schema, counts
# =============================================================
print("Schema:")
df.printSchema()

print(f"Total rows: {df.count()}")

# Row counts per column (non-null)
display(df.summary())

# =============================================================
# 3. Focus on questions - 7a to 7o (Q7A_ART to Q7O_WIFISVC)
#    Use answer key: 1=poor, 2=fair, 3=average, 4=good, 5=excellent
#    0=not applicable, 6=not used/don't know
# =============================================================
# Select relevant columns + target Q13Overall
feature_cols = [
    "Q7A_ART", "Q7B_FOOD", "Q7C_SHOP", "Q7D_INFO", "Q7E_WALK", 
    "Q7F_WIFI", "Q7G_ROAD", "Q7H_PARK", "Q7I_GATE", "Q7J_SAFE", 
    "Q7K_CLEAN", "Q7L_SCREENS", "Q7M_SIGNS", "Q7N_WHOLE", "Q7O_WIFISVC"
]
target_col = "Q13Overall"

df_selected = df.select(feature_cols + [target_col])

# =============================================================
# 4. Get average rating for all constituent ratings
# =============================================================
avg_ratings = df_selected.select([avg(col(c)).alias(f"avg_{c}") for c in feature_cols])
display(avg_ratings)

# Overall average across all
overall_avg = df_selected.select(avg(lit(1)).alias("dummy")).withColumn("overall_avg", 
    avg(*(col(c) for c in feature_cols))).drop("dummy")
print("Overall average rating across all Q7 features:")
display(overall_avg)

# =============================================================
# 5. Infer barchart with percentages and mean of each column
# =============================================================
# For each feature, compute mean and percentage distribution
for c in feature_cols:
    dist = df_selected.groupBy(c).agg(count("*").alias("count"))
    total = dist.select(lit(df_selected.count()).alias("total")).first()["total"]
    dist = dist.withColumn("percentage", round((col("count") / total) * 100, 2))
    mean_val = df_selected.select(avg(c)).first()[0]
    
    print(f"Distribution for {c} (Mean: {mean_val:.2f}):")
    display(dist.orderBy(c))

# =============================================================
# 7. Replace all responses of 0 and 6 with average rating 3
# =============================================================
# Note: Step 6 is ML model description, skipped in code order
for c in feature_cols + [target_col]:
    df_selected = df_selected.withColumn(c, when(col(c).isin(0, 6), 3).otherwise(col(c)))

# Drop rows where target is null or invalid (if any)
df_selected = df_selected.na.drop(subset=[target_col])

# Split train/test
train_df, test_df = df_selected.randomSplit([0.8, 0.2], seed=42)

# =============================================================
# 8. Define ML pipeline - DecisionTreeRegressor, VectorAssembler, 
#    ParamGridBuilder, CrossValidator, RegressionEvaluator (RMSE)
# =============================================================
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

dt = DecisionTreeRegressor(featuresCol="features", labelCol=target_col, seed=42)

pipeline = Pipeline(stages=[assembler, dt])

evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="rmse")

paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [3, 5, 7, 10])
             .addGrid(dt.maxBins, [16, 32, 64])
             .build())

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=5,
    seed=42,
    parallelism=4
)

# =============================================================
# 9. Train the model
# =============================================================
print("Training Decision Tree Regressor with 5-fold CV...")
cv_model = cv.fit(train_df)

# Best model
best_model = cv_model.bestModel
best_dt = best_model.stages[-1]  # DecisionTreeRegressionModel

# Best params
print(f"Best maxDepth: {best_dt.getMaxDepth()}")
print(f"Best maxBins: {best_dt.getMaxBins()}")

# =============================================================
# 10. Display the tree
# =============================================================
print("Decision Tree Model:")
print(best_dt.toDebugString)

# =============================================================
# 11. Make predictions
# =============================================================
predictions = best_model.transform(test_df)

# Show sample
print("Sample Predictions:")
display(predictions.select(feature_cols[:3] + [target_col, "prediction"]).limit(10))

# =============================================================
# 12. Evaluate model
# =============================================================
rmse = evaluator.evaluate(predictions)
mae_evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="mae")
mae = mae_evaluator.evaluate(predictions)
r2_evaluator = RegressionEvaluator(labelCol=target_col, predictionCol="prediction", metricName="r2")
r2 = r2_evaluator.evaluate(predictions)

print(f"\n=== Model Performance on Test Set ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# =============================================================
# 6 & 13. ML model - factors driving overall rating
#         Provide feature importance list
# =============================================================
# Decision Tree feature importances
importances = best_dt.featureImportances.toArray()
feature_importance = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

print("\nFeature Importances (driving factors for overall rating):")
for feature, imp in feature_importance:
    print(f"{feature}: {imp:.4f}")

# Optional: Bar chart visualization
importance_df = spark.createDataFrame(feature_importance, ["Feature", "Importance"])
display(importance_df)