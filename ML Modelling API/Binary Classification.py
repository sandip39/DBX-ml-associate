import mlflow
import mlflow.spark
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials
from hyperopt.pyll import scope
import requests
import json

# Step 1: Load dataset
spark = SparkSession.builder.appName("AdultIncomeClassification").getOrCreate()

# Load from UCI URL (comma-separated, no header)
data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
raw_data = spark.read.csv(data_url, header=False, inferSchema=True, sep=",")

# Define column names from UCI documentation
columns = [
    "age", "workclass", "fnlwgt", "education", "education_num", "marital_status",
    "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss",
    "hours_per_week", "native_country", "income"
]
data = raw_data.toDF(*columns)

# Load test data similarly (skip first line as it's a comment)
test_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
raw_test = spark.read.csv(test_url, header=False, inferSchema=True, sep=",", ignoreLeadingWhiteSpace=True).dropna()
test_data = raw_test.toDF(*columns)  # Note: income in test has '.' at end, clean it below

# Step 2: Data Preprocessing and Feature Engineering
# Handle missing values ('?' in dataset)
categorical_cols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]
numerical_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]

for col_name in categorical_cols:
    data = data.withColumn(col_name, when(col(col_name) == " ?", None).otherwise(col(col_name)))
    test_data = test_data.withColumn(col_name, when(col(col_name) == " ?", None).otherwise(col(col_name)))

# Fill missing with mode (simple imputation)
for col_name in categorical_cols:
    mode = data.groupBy(col_name).count().orderBy("count", ascending=False).first()[0]
    data = data.na.fill(mode, [col_name])
    test_data = test_data.na.fill(mode, [col_name])

# Clean income in test data (remove '.')
test_data = test_data.withColumn("income", when(col("income").endswith("."), col("income").substr(1, length(col("income"))-1)).otherwise(col("income")))

# Index label: >50K as 1, <=50K as 0
label_indexer = StringIndexer(inputCol="income", outputCol="label", stringOrderType="alphabetAsc")  # <=50K:0, >50K:1

# Index and OHE categoricals
indexers = [StringIndexer(inputCol=col, outputCol=col+"_idx", handleInvalid="keep") for col in categorical_cols]
encoders = [OneHotEncoder(inputCol=col+"_idx", outputCol=col+"_ohe") for col in categorical_cols]

# Assemble features
assembler = VectorAssembler(inputCols=numerical_cols + [col+"_ohe" for col in categorical_cols], outputCol="features")

# Split data (use full data for train/val, separate test)
train_full = data.randomSplit([0.8, 0.2], seed=42)
train_data = train_full[0]
val_data = train_full[1]

# Step 3: Code for three classification methods with Hyperopt
# Define evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

def objective_wrapper(model_class, params_space):
    def objective(params):
        with mlflow.start_run(nested=True):
            # Build model
            if model_class == LogisticRegression:
                model = model_class(featuresCol="features", labelCol="label", **params)
            elif model_class == DecisionTreeClassifier:
                model = model_class(featuresCol="features", labelCol="label", **params)
            elif model_class == RandomForestClassifier:
                model = model_class(featuresCol="features", labelCol="label", **params)
            
            # Pipeline
            pipeline = Pipeline(stages=[label_indexer] + indexers + encoders + [assembler, model])
            
            # Fit
            fitted_model = pipeline.fit(train_data)
            
            # Evaluate
            predictions = fitted_model.transform(val_data)
            auc = evaluator.evaluate(predictions)
            
            # Log
            mlflow.log_params(params)
            mlflow.log_metric("auc", auc)
            mlflow.spark.log_model(fitted_model, "model")
            
            return {'loss': -auc, 'status': STATUS_OK}
    
    return objective

# Hyperopt setup
spark_trials = SparkTrials(parallelism=4)  # Distributed

# Logistic Regression
with mlflow.start_run(run_name="LR_Hyperopt"):
    lr_space = {
        'regParam': hp.uniform('regParam', 0.0, 1.0),
        'elasticNetParam': hp.uniform('elasticNetParam', 0.0, 1.0)
    }
    best_lr = fmin(fn=objective_wrapper(LogisticRegression, lr_space), space=lr_space, algo=tpe.suggest, max_evals=10, trials=spark_trials)
    best_lr_auc = -spark_trials.best_trial['result']['loss']

# Decision Tree
with mlflow.start_run(run_name="DT_Hyperopt"):
    dt_space = {
        'maxDepth': scope.int(hp.quniform('maxDepth', 2, 10, 1)),
        'maxBins': scope.int(hp.quniform('maxBins', 16, 64, 8))
    }
    best_dt = fmin(fn=objective_wrapper(DecisionTreeClassifier, dt_space), space=dt_space, algo=tpe.suggest, max_evals=10, trials=spark_trials)
    best_dt_auc = -spark_trials.best_trial['result']['loss']

# Random Forest
with mlflow.start_run(run_name="RF_Hyperopt"):
    rf_space = {
        'numTrees': scope.int(hp.quniform('numTrees', 10, 100, 10)),
        'maxDepth': scope.int(hp.quniform('maxDepth', 2, 10, 1))
    }
    best_rf = fmin(fn=objective_wrapper(RandomForestClassifier, rf_space), space=rf_space, algo=tpe.suggest, max_evals=10, trials=spark_trials)
    best_rf_auc = -spark_trials.best_trial['result']['loss']

# Step 4: Choose best model
aucs = {"LR": best_lr_auc, "DT": best_dt_auc, "RF": best_rf_auc}
best_model_name = max(aucs, key=aucs.get)
best_params = locals()[f"best_{best_model_name.lower()}"]
best_auc = aucs[best_model_name]
print(f"Best model: {best_model_name} with AUC: {best_auc}")

# Step 5: Provide parameters for best model
print(f"Best parameters: {best_params}")

# Retrain best model on full train data
with mlflow.start_run(run_name=f"Best_{best_model_name}"):
    model_class = globals()[f"{best_model_name}Classifier" if best_model_name != "LogisticRegression" else "LogisticRegression"]
    best_model = model_class(featuresCol="features", labelCol="label", **best_params)
    best_pipeline = Pipeline(stages=[label_indexer] + indexers + encoders + [assembler, best_model])
    fitted_best = best_pipeline.fit(data)  # Full data for final model
    mlflow.log_params(best_params)
    mlflow.log_metric("train_auc", evaluator.evaluate(fitted_best.transform(data)))
    model_uri = mlflow.spark.log_model(fitted_best, "best_model").model_uri

# Step 6: Register model
registered_model_name = "AdultIncomeBest"
mlflow.register_model(model_uri, registered_model_name)

# Step 7: Use it for batch predictions
batch_predictions = fitted_best.transform(test_data)
batch_predictions.select("features", "prediction", "probability").show(5)
test_auc = evaluator.evaluate(batch_predictions)
print(f"Test AUC: {test_auc}")

# Step 8: Register real-time model (for serving)
# Note: In Databricks, enable serving via UI or API. Here, we log as pyfunc for serving.
class IncomePyFunc(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        # Assume model_input is a DataFrame with columns matching input schema
        return self.model.transform(model_input).select("prediction").collect()

pyfunc_model = IncomePyFunc(fitted_best)
input_example = test_data.limit(1).drop("income", "label")
signature = infer_signature(input_example.toPandas(), fitted_best.transform(input_example).select("prediction").toPandas())
mlflow.pyfunc.log_model("pyfunc_model", python_model=pyfunc_model, input_example=input_example.toPandas(), signature=signature)
pyfunc_uri = mlflow.get_artifact_uri("pyfunc_model")

# Register pyfunc version
mlflow.register_model(pyfunc_uri, "AdultIncomeServing")

# To create serving endpoint, use Databricks UI: Machine Learning > Model Registry > Model > Create Endpoint
# Or use Databricks SDK/REST API (example below requires databricks-sdk installed)
# from databricks.sdk import WorkspaceClient
# w = WorkspaceClient()
# w.serving_endpoints.create(name="income-endpoint", config={"served_models": [{"model_name": "AdultIncomeServing", "model_version": "latest", "workload_size": "Small"}]} )

# Step 9: Use it for real-time predictions
# Assuming endpoint created at /serving-endpoints/income-endpoint/invocations
# Token: Generate from Databricks UI
databricks_token = "your_databricks_token"
endpoint_url = "https://your-databricks-instance.databricks.com/serving-endpoints/income-endpoint/invocations"

# Sample input (JSON format matching input schema)
sample_input = {
    "dataframe_split": {
        "columns": columns[:-1],  # Exclude income
        "data": [[39, "State-gov", 77516, "Bachelors", 13, "Never-married", "Adm-clerical", "Not-in-family", "White", "Male", 2174, 0, 40, "United-States"]]
    }
}

headers = {
    "Authorization": f"Bearer {databricks_token}",
    "Content-Type": "application/json"
}

response = requests.post(endpoint_url, headers=headers, data=json.dumps(sample_input))
print(response.json())  # {'predictions': [0.0] or similar}

# Step 10: Provide code for A/B experimentation
# For A/B: Train/register two models (e.g., LR and RF as champion/challenger)
# Create parent experiment
mlflow.set_experiment("/AB_Experiments/IncomeModels")

# Log champion (e.g., LR)
with mlflow.start_run(run_name="Champion_LR"):
    # Reuse from above, log model, assign alias @champion in registry UI

# Log challenger (e.g., RF)
with mlflow.start_run(run_name="Challenger_RF"):
    # Similar

# Deploy to single endpoint with traffic split (via UI or API)
# Example API (databricks-sdk):
# w.serving_endpoints.update_config(
#     name="ab-income-endpoint",
#     traffic_config={
#         "routes": [
#             {"served_model_name": "AdultIncomeServing@champion", "traffic_percentage": 70},
#             {"served_model_name": "AdultIncomeServing@challenger", "traffic_percentage": 30}
#         ]
#     }
# )

# Monitor: Log predictions/feedback to Delta table, compute metrics
# Example: Streaming feedback
# feedback_stream = spark.readStream.table("feedback_table")  # With model_version, prediction, ground_truth
# Then aggregate metrics per model_version

# Step 11: Use routes
# In serving endpoint, routes define traffic splits for A/B (as above).
# Query the A/B endpoint similarly to Step 9; traffic routed automatically.
# To force a route (if configured), add header or param, but default is probabilistic split.