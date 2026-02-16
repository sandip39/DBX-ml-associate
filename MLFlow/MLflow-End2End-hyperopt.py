import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from functools import partial

# --- 1. Load Data and Feature Engineering ---

# Load red and white wine data. Replace catalog and schema names if using Unity Catalog.
# Includes fallback for local testing.
try:
    red_wine = spark.table("`catalog_name`.`schema_name`.winequality_red").toPandas()
    white_wine = spark.table("`catalog_name`.`schema_name`.winequality_white").toPandas()
except Exception as e:
    print(f"Could not load data from Unity Catalog: {e}. Attempting to load from UCI repository.")
    data_url_red = "https://archive.ics.uci.edu"
    data_url_white = "https://archive.ics.uci.uci.edu"
    red_wine = pd.read_csv(data_url_red, sep=';')
    white_wine = pd.read_csv(data_url_white, sep=';')

# Add 'is_red' feature, combine data, rename columns, and create binary target
red_wine['is_red'] = 1
white_wine['is_red'] = 0
data = pd.concat([red_wine, white_wine], axis=0)
data.rename(columns=lambda x: x.replace(' ', '_'), inplace=True)
data['high_quality'] = (data['quality'] >= 7).astype(int)

X = data.drop(['quality', 'high_quality'], axis=1)
y = data['high_quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Define Objective Function for Hyperopt ---

def objective_function(params):
    # Ensure params have correct types
    params['max_depth'] = int(params['max_depth'])
    params['min_child_weight'] = int(params['min_child_weight'])

    with mlflow.start_run(nested=True):
        # Define and train pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb_classifier', xgb.XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss'))
        ])
        pipeline.fit(X_train, y_train)

        # Evaluate model and log metrics
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_prob)
        mlflow.log_metric("test_auc", auc_score)

        # Hyperopt minimizes loss
        loss = -auc_score
        return {'loss': loss, 'status': STATUS_OK}

# --- 3. Define Search Space and Run Hyperopt ---

search_space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
    'max_depth': hp.quniform('max_depth', 3, 10, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 6, 1),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
}

# Use SparkTrials for parallel tuning
spark_trials = SparkTrials(parallelism=4)

with mlflow.start_run(run_name='Hyperopt_XGBoost_Tuning'):
    best_params = fmin(
        fn=objective_function,
        space=search_space,
        algo=tpe.suggest,
        max_evals=25, # Number of trials
        trials=spark_trials
    )

    # Convert best params to proper types
    best_params['max_depth'] = int(best_params['max_depth'])
    best_params['min_child_weight'] = int(best_params['min_child_weight'])

    # --- 4. Train Final Model with Best Parameters and Register ---

    # Train final model with autologging
    mlflow.xgboost.autolog(silent=True)
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('xgb_classifier', xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss'))
    ])
    final_pipeline.fit(X_train, y_train)

    # Register the model. Use Unity Catalog naming if desired (e.g., "ml.production.wine_quality_model")
    model_name = "wine_quality_model"
    mlflow.register_model(
        model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
        name=model_name
    )

    print(f"Best parameters found: {best_params}")
    print(f"Registered model as: {model_name}")


from mlflow.tracking import MlflowClient

client = MlflowClient()
model_name = "wine_quality_model"

# Transition to Production stage (for Workspace Registry)
# Use aliases for Unity Catalog
try:
    client.transition_model_version_stage(
        name=model_name,
        version=1, # Specify correct version
        stage="Production"
    )
    print(f"Model {model_name} version 1 transitioned to Production")
except Exception as e:
    print(f"Could not transition model stage (maybe using Unity Catalog?): {e}")

# Example using aliases for Unity Catalog
# client.set_registered_model_alias(name=model_name, alias="Champion", version=1)
