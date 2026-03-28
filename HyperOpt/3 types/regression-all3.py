# =============================================================================
# META-OPTIMIZATION FOR HOUSE PRICE PREDICTION
# Tuning Method (optuna / hyperopt / spark_ml) is a hyperparameter to optimize
# =============================================================================

%pip install -q scikit-learn hyperopt optuna mlflow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Spark ML imports
from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler
from pyspark.ml.regression import RandomForestRegressor as SparkRandomForest
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
mlflow.set_experiment("/Users/your.email@domain.com/Housing_Meta_Tuning_Method")

# =============================================================================
# LOAD CALIFORNIA HOUSING DATA
# =============================================================================
print("Loading California Housing dataset...")
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Use full dataset (~20k rows) or sample if needed
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)} | Test samples: {len(X_test)}")

# =============================================================================
# META OBJECTIVE: Optimize tuning method + hyperparameters
# =============================================================================
def meta_objective(trial):
    # Tuning method is now a hyperparameter to optimize
    tuning_method = trial.suggest_categorical("tuning_method", ["optuna", "hyperopt", "spark_ml"])
    
    with mlflow.start_run(nested=True, run_name=f"MetaTrial_{trial.number}_{tuning_method}") as child_run:
        mlflow.log_param("tuning_method", tuning_method)
        
        rmse = None
        best_params = None

        # ====================== INNER TUNING BASED ON CHOSEN METHOD ======================
        if tuning_method == "optuna":
            import optuna
            def inner_obj(t):
                n_est = t.suggest_int("n_estimators", 50, 300, step=50)
                max_d = t.suggest_int("max_depth", 5, 25)
                min_split = t.suggest_int("min_samples_split", 2, 10)
                
                model = RandomForestRegressor(n_estimators=n_est, max_depth=max_d, 
                                              min_samples_split=min_split, random_state=42, n_jobs=-1)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                return mean_squared_error(y_test, y_pred, squared=False)
            
            inner_study = optuna.create_study(direction="minimize")
            inner_study.optimize(inner_obj, n_trials=10)
            rmse = inner_study.best_value
            best_params = inner_study.best_params

        elif tuning_method == "hyperopt":
            from hyperopt import fmin, tpe, hp, STATUS_OK, SparkTrials
            from hyperopt.pyll import scope
            
            def inner_obj(params):
                model = RandomForestRegressor(
                    n_estimators=int(params["n_estimators"]),
                    max_depth=int(params["max_depth"]),
                    min_samples_split=int(params["min_samples_split"]),
                    random_state=42, n_jobs=-1
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                rmse_val = mean_squared_error(y_test, y_pred, squared=False)
                return {"loss": rmse_val, "status": STATUS_OK}
            
            space = {
                "n_estimators": scope.int(hp.quniform("n_estimators", 50, 300, 50)),
                "max_depth": scope.int(hp.quniform("max_depth", 5, 25, 1)),
                "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1))
            }
            
            spark_trials = SparkTrials(parallelism=4, timeout=600)
            best = fmin(inner_obj, space, tpe.suggest, max_evals=10, trials=spark_trials)
            rmse = spark_trials.best_trial["result"]["loss"]
            best_params = best

        elif tuning_method == "spark_ml":
            # Convert to Spark DataFrame
            train_df = spark.createDataFrame(pd.concat([X_train, y_train], axis=1))
            
            feature_cols = X_train.columns.tolist()
            assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
            scaler = SparkScaler(inputCol="features", outputCol="scaled_features")
            rf = SparkRandomForest(featuresCol="scaled_features", labelCol="MedHouseVal")
            
            pipeline = Pipeline(stages=[assembler, scaler, rf])
            
            paramGrid = ParamGridBuilder() \
                .addGrid(rf.numTrees, [50, 100, 150]) \
                .addGrid(rf.maxDepth, [5, 10, 15]) \
                .build()
            
            evaluator = RegressionEvaluator(labelCol="MedHouseVal", predictionCol="prediction", metricName="rmse")
            
            cv = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid,
                                evaluator=evaluator, numFolds=3, parallelism=4)
            
            cvModel = cv.fit(train_df)
            predictions = cvModel.bestModel.transform(train_df)
            rmse = evaluator.evaluate(predictions)
            best_params = {
                "numTrees": cvModel.bestModel.stages[2].getNumTrees(),
                "maxDepth": cvModel.bestModel.stages[2].getMaxDepth()
            }

        # Log results of this meta-trial
        mlflow.log_param("best_params", best_params)
        mlflow.log_metric("rmse", rmse)
        
        return rmse   # We want to minimize RMSE

# =============================================================================
# RUN META OPTIMIZATION
# =============================================================================
with mlflow.start_run(run_name="Meta_Optimization_Tuning_Method_Housing") as parent_run:
    mlflow.log_param("meta_n_trials", 9)
    
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(meta_objective, n_trials=9)   # Number of times we try different tuning methods

    # Final Summary
    print("\n" + "="*80)
    print("META OPTIMIZATION RESULTS - HOUSE PRICE PREDICTION")
    print("="*80)
    print(f"Best Tuning Method     : {study.best_params['tuning_method']}")
    print(f"Best RMSE              : {study.best_value:.4f}")
    print(f"Best Meta Trial        : {study.best_trial.number}")
    print("="*80)

print("\n✅ Meta-optimization completed!")
print("Check MLflow UI for the parent run with nested child runs (one per meta-trial).")