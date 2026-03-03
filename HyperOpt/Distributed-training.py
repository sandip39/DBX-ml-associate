# =============================================
# DISTRIBUTED TRAINING WITH MLLIB IN DATABRICKS
# MNIST Dataset (libsvm format) – Decision Tree Classifier
# =============================================

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow.pyspark.ml  # for autologging (Databricks-specific)

spark = SparkSession.builder.appName("MNIST_DecisionTree").getOrCreate()

# Step 1: Load MNIST dataset in libsvm format
# Assume files are uploaded to DBFS (e.g., via Databricks UI: Data → Add Data → Upload)
# MNIST libsvm from: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html
# Download mnist.scale (train) and mnist.scale.t (test), rename to .libsvm if needed
train_path = "/FileStore/mnist_train.libsvm"  # adjust to your DBFS path
test_path = "/FileStore/mnist_test.libsvm"    # adjust to your DBFS path

# Load full training and test data (libsvm format: label followed by index:value pairs)
df_full_train = spark.read.format("libsvm").load(train_path).cache()
df_test = spark.read.format("libsvm").load(test_path).cache()

print("Full train count:", df_full_train.count())
print("Test count:", df_test.count())

# Step 2: Randomly split full training data into train/validation (80/20) with seed
train_df, val_df = df_full_train.randomSplit([0.8, 0.2], seed=42)

print("Train split count:", train_df.count())
print("Validation split count:", val_df.count())

# Step 3: Create function to train model (wrapper for hyper-tuning later)
# Uses StringIndexer (if labels are string; but MNIST labels are numeric 0-9 → optional but included)
# Multiclass evaluator for F1
def train_model(train_data, val_data, min_instances_per_node=1, max_bins=32):
    # StringIndexer for labels (handles if needed; for numeric, it passes through)
    label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel", handleInvalid="skip")
    
    # DecisionTreeClassifier
    dt = DecisionTreeClassifier(
        labelCol="indexedLabel",
        featuresCol="features",
        minInstancesPerNode=min_instances_per_node,
        maxBins=max_bins
    )
    
    # Pipeline
    pipeline = Pipeline(stages=[label_indexer, dt])
    
    # Train
    model = pipeline.fit(train_data)
    
    # Predict on validation
    predictions = model.transform(val_data)
    
    # Multiclass evaluator for F1 (weighted)
    evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel",
        predictionCol="prediction",
        metricName="f1"
    )
    
    f1_score = evaluator.evaluate(predictions)
    
    return model, f1_score

# Step 4: Load mlflow.pyspark.ml with try-except (for autologging in Databricks Runtime with MLflow)
try:
    import mlflow.pyspark.ml
    mlflow.pyspark.ml.autolog()
    print("MLflow PySpark ML autologging enabled.")
except ImportError:
    print("MLflow PySpark ML not available – skipping autologging.")

# Step 5: Define function train_tree (for hyper-tuning)
def train_tree(min_instances_per_node, max_bins):
    # Train and get F1 from validation
    _, f1_score = train_model(train_df, val_df, min_instances_per_node, max_bins)
    return f1_score

# Step 6: Train with some parameters (example: minInstancesPerNode=5, maxBins=64)
with mlflow.start_run(run_name="DecisionTree_MNIST"):
    model, f1 = train_model(train_df, val_df, min_instances_per_node=5, max_bins=64)
    
    # Log params and metrics manually if autolog fails
    mlflow.log_param("min_instances_per_node", 5)
    mlflow.log_param("max_bins", 64)
    mlflow.log_metric("val_f1", f1)
    
    # Optional: evaluate on test
    test_preds = model.transform(df_test)
    test_evaluator = MulticlassClassificationEvaluator(
        labelCol="indexedLabel",
        predictionCol="prediction",
        metricName="f1"
    )
    test_f1 = test_evaluator.evaluate(test_preds)
    mlflow.log_metric("test_f1", test_f1)
    
    print(f"Validation F1: {f1:.4f}")
    print(f"Test F1: {test_f1:.4f}")

# For hyper-tuning later: you can use Hyperopt or ParamGridSearch with CrossValidator
# Example stub:
# from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
# paramGrid = ParamGridBuilder() \
#     .addGrid(dt.minInstancesPerNode, [1, 5, 10]) \
#     .addGrid(dt.maxBins, [32, 64, 128]) \
#     .build()
# cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
# cvModel = cv.fit(train_df)