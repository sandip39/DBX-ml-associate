# =============================================
# DISTRIBUTED DECISION TREE TUNING WITH HYPEROPT + SPARK MLlib
# MNIST (libsvm format) – Databricks Notebook
# =============================================

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ── Hyperopt imports ────────────────────────────────────────────────────────
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, SparkTrials
import mlflow
import numpy as np

spark = SparkSession.builder.appName("MNIST_DT_HyperOpt").getOrCreate()

# ── 1. Load MNIST in libsvm format ──────────────────────────────────────────
# Adjust paths to your uploaded files in DBFS
train_path = "/FileStore/mnist/mnist.scale"     # training file
test_path  = "/FileStore/mnist/mnist.scale.t"   # test file

df_full = spark.read.format("libsvm").load(train_path).cache()
df_test  = spark.read.format("libsvm").load(test_path).cache()

print("Full train rows:", df_full.count())
print("Test rows:", df_test.count())

# ── 2. Split train → train + validation ─────────────────────────────────────
train_df, val_df = df_full.randomSplit([0.8, 0.2], seed=42)

print("Train split:", train_df.count())
print("Val split:  ", val_df.count())

# ── 3. Evaluation setup ─────────────────────────────────────────────────────
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel",
    predictionCol="prediction",
    metricName="f1"
)

# ── 4. Objective function to minimize (-F1 score) ───────────────────────────
def objective(params):
    try:
        # Log parameters to MLflow (optional but recommended)
        mlflow.log_params(params)

        # Build pipeline
        label_indexer = StringIndexer(
            inputCol="label",
            outputCol="indexedLabel",
            handleInvalid="skip"
        )

        dt = DecisionTreeClassifier(
            labelCol="indexedLabel",
            featuresCol="features",
            minInstancesPerNode=int(params['minInstancesPerNode']),
            maxBins=int(params['maxBins']),
            maxDepth=int(params['maxDepth']),
            impurity=params['impurity']
        )

        pipeline = Pipeline(stages=[label_indexer, dt])

        # Train
        model = pipeline.fit(train_df)

        # Predict on validation
        predictions = model.transform(val_df)

        # Evaluate
        f1 = evaluator.evaluate(predictions)

        # Log metric
        mlflow.log_metric("val_f1", f1)

        return {
            'loss': -f1,           # minimize → negative F1
            'status': STATUS_OK
        }

    except Exception as e:
        print(f"Error in trial: {str(e)}")
        return {
            'loss': 0.0,
            'status': STATUS_OK   # or STATUS_FAIL to skip bad trials
        }

# ── 5. Define search space ──────────────────────────────────────────────────
search_space = {
    'minInstancesPerNode': hp.quniform('minInstancesPerNode', 1, 50, 1),
    'maxBins':             hp.quniform('maxBins', 16, 128, 8),
    'maxDepth':            hp.quniform('maxDepth', 5, 30, 1),
    'impurity':            hp.choice('impurity', ['gini', 'entropy'])
}

# ── 6. Hyperopt tuning ──────────────────────────────────────────────────────
max_evals = 40   # adjust based on cluster size & time budget

# Use SparkTrials for distributed execution (recommended in Databricks)
spark_trials = SparkTrials(
    parallelism=4,          # adjust based on cluster cores
    timeout=60*60*2         # 2 hours max (in seconds)
)

print(f"Starting Hyperopt tuning ({max_evals} trials)...")

with mlflow.start_run(run_name="DecisionTree_HyperOpt_MNIST"):
    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,           # Tree-structured Parzen Estimator
        max_evals=max_evals,
        trials=spark_trials,
        rstate=np.random.default_rng(42)
    )

# ── 7. Print best results ───────────────────────────────────────────────────
print("\n" + "="*70)
print("HYPEROPT TUNING RESULTS")
print("="*70)

best_f1 = -spark_trials.best_trial['result']['loss']
best_params = spark_trials.best_trial['misc']['vals']

print(f"Best validation F1:      {best_f1:.4f}")
print("Best parameters:")
for param, value in best_params.items():
    # Clean up the list format from hyperopt
    cleaned_value = value[0] if isinstance(value, list) else value
    if param == 'impurity':
        cleaned_value = ['gini', 'entropy'][int(cleaned_value)]
    elif param in ['minInstancesPerNode', 'maxBins', 'maxDepth']:
        cleaned_value = int(cleaned_value)
    print(f"  {param:20} = {cleaned_value}")

print("-"*70)

# Optional: retrain final model with best params on full train data
print("\nRetraining final model with best parameters on full training set...")

final_model, _ = train_model(
    df_full, 
    val_df,  # we don't use val score here, just for compatibility
    min_instances_per_node=int(best_params['minInstancesPerNode'][0]),
    max_bins=int(best_params['maxBins'][0])
)

print("Final model trained.")