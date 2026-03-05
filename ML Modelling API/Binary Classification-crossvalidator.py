# Modified version using Spark ML built-in CrossValidator + ParamGridBuilder
# with numFolds = 5 (5-fold cross-validation) instead of Hyperopt
# We tune three models: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
# Then compare them and select/register/deploy the best one

import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, length
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# ────────────────────────────────────────────────
# Step 1: Load dataset (UCI Adult / Census Income)
# ────────────────────────────────────────────────
spark = SparkSession.builder.appName("AdultIncome_CV_ParamGrid").getOrCreate()

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
raw_data = spark.read.csv(data_url, header=False, inferSchema=True, sep=",")

columns = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]
data = raw_data.toDF(*columns)

# Test set
test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
raw_test = spark.read.csv(test_url, header=False, inferSchema=True, sep=",")
test_data = raw_test.toDF(*columns)

# Clean test income (has '.' at end sometimes)
test_data = test_data.withColumn("income", when(col("income").endswith("."), col("income").substr(1, length(col("income"))-1)).otherwise(col("income")))

# ────────────────────────────────────────────────
# Step 2: Preprocessing & Feature Engineering
# ────────────────────────────────────────────────
categorical_cols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
numerical_cols   = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

# Replace ' ?' with null
for c in categorical_cols:
    data     = data.withColumn(c, when(col(c) == " ?", None).otherwise(col(c)))
    test_data = test_data.withColumn(c, when(col(c) == " ?", None).otherwise(col(c)))

# Simple mode imputation for categoricals
for c in categorical_cols:
    mode_val = data.groupBy(c).count().orderBy("count", ascending=False).first()[0]
    data     = data.na.fill(mode_val, [c])
    test_data = test_data.na.fill(mode_val, [c])

# Label: <=50K → 0, >50K → 1
label_indexer = StringIndexer(inputCol="income", outputCol="label", stringOrderType="alphabetAsc")

# String → Index → OneHot
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical_cols]
encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_ohe") for c in categorical_cols]

assembler = VectorAssembler(
    inputCols = numerical_cols + [f"{c}_ohe" for c in categorical_cols],
    outputCol = "features"
)

# Split train / val (we'll use CV on train, final eval on test)
train_df, val_df = data.randomSplit([0.8, 0.2], seed=42)

# Evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

# ────────────────────────────────────────────────
# Step 3: Define pipelines + CrossValidator (k=5) for each model
# ────────────────────────────────────────────────

def create_cv_pipeline(model, param_grid):
    pipeline = Pipeline(stages=[label_indexer] + indexers + encoders + [assembler, model])
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=5,           # ← 5-fold CV as requested
        seed=42,
        parallelism=4         # adjust based on cluster
    )
    return cv

# ── Logistic Regression ───────────────────────────────
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=100)
lr_grid = (ParamGridBuilder()
           .addGrid(lr.regParam, [0.001, 0.01, 0.1])
           .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
           .build())
cv_lr = create_cv_pipeline(lr, lr_grid)

# ── Decision Tree ─────────────────────────────────────
dt = DecisionTreeClassifier(featuresCol="features", labelCol="label")
dt_grid = (ParamGridBuilder()
           .addGrid(dt.maxDepth, [5, 10, 15, 20])
           .addGrid(dt.maxBins, [32, 64])
           .build())
cv_dt = create_cv_pipeline(dt, dt_grid)

# ── Random Forest ─────────────────────────────────────
rf = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42)
rf_grid = (ParamGridBuilder()
           .addGrid(rf.numTrees, [50, 100, 150])
           .addGrid(rf.maxDepth, [5, 10, 15])
           .addGrid(rf.subsamplingRate, [0.8, 1.0])
           .build())
cv_rf = create_cv_pipeline(rf, rf_grid)

# ────────────────────────────────────────────────
# Train & evaluate each with 5-fold CV + log to MLflow
# ────────────────────────────────────────────────

models = {"LogisticRegression": cv_lr, "DecisionTree": cv_dt, "RandomForest": cv_rf}
best_models = {}
metrics = {}

for name, cv in models.items():
    with mlflow.start_run(run_name=f"CV_{name}_5fold"):
        cv_model = cv.fit(train_df)
        best_sub_model = cv_model.bestModel
        
        # Validation AUC (on hold-out val set)
        val_pred = best_sub_model.transform(val_df)
        val_auc = evaluator.evaluate(val_pred)
        
        # Average CV metric (from folds)
        avg_cv_auc = max(cv_model.avgMetrics)   # since higher is better for AUC
        
        mlflow.log_metric("val_auc", val_auc)
        mlflow.log_metric("cv_avg_auc", avg_cv_auc)
        mlflow.spark.log_model(best_sub_model, "model")
        
        best_models[name] = best_sub_model
        metrics[name] = val_auc
        
        print(f"{name} → CV avg AUC: {avg_cv_auc:.4f} | Val AUC: {val_auc:.4f}")

# ────────────────────────────────────────────────
# Step 4 & 5: Choose best model + get parameters
# ────────────────────────────────────────────────
best_name = max(metrics, key=metrics.get)
best_model = best_models[best_name]
best_val_auc = metrics[best_name]

print(f"\nBest model: {best_name} with validation AUC = {best_val_auc:.4f}")

# Extract best params
best_stages = best_model.stages
best_classifier = best_stages[-1]

if best_name == "LogisticRegression":
    print("Best params:", {
        "regParam": best_classifier.getRegParam(),
        "elasticNetParam": best_classifier.getElasticNetParam()
    })
elif best_name == "DecisionTree":
    print("Best params:", {
        "maxDepth": best_classifier.getMaxDepth(),
        "maxBins": best_classifier.getMaxBins()
    })
else:  # RandomForest
    print("Best params:", {
        "numTrees": best_classifier.getNumTrees,
        "maxDepth": best_classifier.getMaxDepth(),
        "subsamplingRate": best_classifier.getSubsamplingRate()
    })

# ────────────────────────────────────────────────
# Retrain on full training data (no split) for production
# ────────────────────────────────────────────────
with mlflow.start_run(run_name=f"Final_{best_name}"):
    final_model = best_classifier  # already has best params
    final_pipeline = Pipeline(stages=[label_indexer] + indexers + encoders + [assembler, final_model])
    final_fitted = final_pipeline.fit(data)  # full data
    
    test_pred = final_fitted.transform(test_data)
    test_auc = evaluator.evaluate(test_pred)
    mlflow.log_metric("test_auc", test_auc)
    
    model_uri = mlflow.spark.log_model(final_fitted, "best_model").model_uri

# ────────────────────────────────────────────────
# Step 6: Register model
# ────────────────────────────────────────────────
registered_name = "AdultIncome_Classifier_CV"
mlflow.register_model(model_uri, registered_name)

# ────────────────────────────────────────────────
# Step 7: Batch predictions
# ────────────────────────────────────────────────
batch_results = final_fitted.transform(test_data)
batch_results.select("age", "education", "hours_per_week", "prediction", "probability").show(10, truncate=False)

# ────────────────────────────────────────────────
# Steps 8–11: Real-time serving & A/B (same as previous example)
# ────────────────────────────────────────────────
# See previous code for:
#   • pyfunc wrapper + logging for serving
#   • Registering for Model Serving
#   • Creating endpoint via UI or SDK
#   • Invoking real-time predictions via REST
#   • Setting up A/B with traffic routes (@champion / @challenger)

print(f"\nTest AUC (best model): {test_auc:.4f}")
print("Done. Model registered → check MLflow → Model Registry")