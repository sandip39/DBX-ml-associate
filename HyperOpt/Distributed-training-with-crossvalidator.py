import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# ────────────────────────────────────────────────
#  Spark Session (already exists in Databricks)
# ────────────────────────────────────────────────
spark = SparkSession.builder.appName("MNIST_DT_with_MLflow").getOrCreate()

# ────────────────────────────────────────────────
# 1. Load & cache data (update paths to your actual location)
# ────────────────────────────────────────────────
train_full = spark.read.format("libsvm") \
    .load("/dbfs/mnist/mnist_train.libsvm") \
    .cache()

test_data = spark.read.format("libsvm") \
    .load("/dbfs/mnist/mnist_test.libsvm") \
    .cache()

# Split with seed
train_data, val_data = train_full.randomSplit([0.8, 0.2], seed=42)
train_data.cache()
val_data.cache()

# ────────────────────────────────────────────────
# 2. Define the tuning & training logic with MLflow
# ────────────────────────────────────────────────
def train_and_log_with_mlflow():
    # Start MLflow run (this is the key part you asked for)
    with mlflow.start_run(run_name="DecisionTree_MNIST_CV") as run:
        # ── Logging basic run info ───────────────────────────────
        mlflow.log_param("train_split_ratio", 0.8)
        mlflow.log_param("validation_split_ratio", 0.2)
        mlflow.log_param("random_seed", 42)
        mlflow.set_tag("dataset", "MNIST libsvm")
        mlflow.set_tag("model_type", "DecisionTreeClassifier + CV")

        # ── Components ───────────────────────────────────────────
        label_indexer = StringIndexer(
            inputCol="label",
            outputCol="indexedLabel",
            handleInvalid="skip"
        )

        dt = DecisionTreeClassifier(
            labelCol="indexedLabel",
            featuresCol="features"
        )

        pipeline = Pipeline(stages=[label_indexer, dt])

        # Evaluator
        evaluator = MulticlassClassificationEvaluator(
            labelCol="indexedLabel",
            predictionCol="prediction",
            metricName="accuracy"
        )

        # Parameter grid
        paramGrid = (ParamGridBuilder()
                     .addGrid(dt.maxDepth, [5, 10, 15, 20])
                     .addGrid(dt.maxBins,  [32, 64, 128])
                     .build())

        # CrossValidator
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=paramGrid,
            evaluator=evaluator,
            numFolds=3,
            seed=42,
            parallelism=4          # adjust based on cluster size
        )

        # ── Fit CrossValidator ───────────────────────────────────
        cv_model = cv.fit(train_data)

        # Best model
        best_model = cv_model.bestModel
        best_pipeline = best_model  # it's already a PipelineModel

        # Get best hyperparameters
        best_max_depth = best_model.stages[-1].getMaxDepth()
        best_max_bins  = best_model.stages[-1].getMaxBins()

        # Log best parameters
        mlflow.log_param("best_maxDepth", best_max_depth)
        mlflow.log_param("best_maxBins", best_max_bins)

        # ── Evaluate on validation set ───────────────────────────
        val_predictions = best_model.transform(val_data)
        val_accuracy = evaluator.evaluate(val_predictions)

        mlflow.log_metric("val_accuracy", val_accuracy)
        mlflow.log_metric("cv_num_folds", 3)

        # ── Evaluate on test set ─────────────────────────────────
        test_predictions = best_model.transform(test_data)
        test_accuracy = evaluator.evaluate(test_predictions)

        mlflow.log_metric("test_accuracy", test_accuracy)

        # ── Log the best Spark ML model ──────────────────────────
        mlflow.spark.log_model(
            spark_model=best_model,
            artifact_path="best-decision-tree-model",
            registered_model_name="MNIST_DecisionTree_Best"   # optional
        )

        # Optional: log average metrics from all folds
        avg_metrics = cv_model.avgMetrics
        for i, acc in enumerate(avg_metrics):
            mlflow.log_metric(f"fold_{i+1}_accuracy", acc)

        print(f"Best maxDepth     : {best_max_depth}")
        print(f"Best maxBins      : {best_max_bins}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Test accuracy      : {test_accuracy:.4f}")
        print(f"MLflow run ID      : {run.info.run_id}")

        return best_model, run.info.run_id


# ────────────────────────────────────────────────
# Run everything inside one MLflow run
# ────────────────────────────────────────────────
best_dt_model, run_id = train_and_log_with_mlflow()

print(f"\nDone. View run → {mlflow.get_artifact_uri()}")
print(f"Or go to MLflow UI → /mlflow/runs/{run_id}")