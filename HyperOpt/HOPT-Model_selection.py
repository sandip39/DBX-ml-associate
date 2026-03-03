# =============================================
# HYPERPARAMETER OPTIMIZATION + MODEL COMPARISON
# California Housing → Classification (3 classes)
# Databricks Notebook – Python Cell
# =============================================

# Step 1: Imports
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import mlflow
import mlflow.sklearn

# Load dataset
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y_continuous = housing.target

# Step 2: Feature Engineering
# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert regression target → classification (3 balanced classes)
y = pd.qcut(y_continuous, q=3, labels=[0, 1, 2]).astype(int)

print("Class distribution:\n", pd.Series(y).value_counts(normalize=True))

# ────────────────────────────────────────────────────────────────
# Define objective function factory (model-specific)
# ────────────────────────────────────────────────────────────────

def create_objective(model_name, model_class, param_space):
    def objective(params):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_name)
            
            # Instantiate model with current params
            if model_name == "SVC":
                model = model_class(
                    C=params['C'],
                    gamma=params['gamma'],
                    kernel=params['kernel'],
                    random_state=42,
                    max_iter=2000,           # prevent very long runs
                    probability=False
                )
            elif model_name == "RandomForest":
                model = model_class(
                    n_estimators=int(params['n_estimators']),
                    max_depth=int(params['max_depth']),
                    min_samples_split=int(params['min_samples_split']),
                    random_state=42,
                    n_jobs=-1
                )
            elif model_name == "LogisticRegression":
                model = model_class(
                    C=params['C'],
                    penalty=params['penalty'],
                    solver='lbfgs' if params['penalty'] == 'l2' else 'liblinear',
                    max_iter=1000,
                    random_state=42
                )
            
            # 5-fold CV accuracy
            score = cross_val_score(
                model, X_scaled, y,
                cv=5, scoring='accuracy', n_jobs=-1
            ).mean()
            
            mlflow.log_metric("cv_accuracy", score)
            
            return {'loss': -score, 'status': STATUS_OK}
    
    return objective

# ────────────────────────────────────────────────────────────────
# Search spaces for each model
# ────────────────────────────────────────────────────────────────

search_spaces = {
    "RandomForest": {
        'n_estimators':     hp.quniform('n_estimators', 50, 300, 25),
        'max_depth':        hp.quniform('max_depth', 5, 40, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 20, 1)
    },
    "SVC": {
        'C':     hp.loguniform('C', np.log(1e-3), np.log(1e3)),
        'gamma': hp.loguniform('gamma', np.log(1e-4), np.log(1e1)),
        'kernel': hp.choice('kernel', ['rbf', 'poly', 'sigmoid'])   # linear often too slow
    },
    "LogisticRegression": {
        'C':      hp.loguniform('C', np.log(1e-4), np.log(1e2)),
        'penalty': hp.choice('penalty', ['l1', 'l2'])
    }
}

# ────────────────────────────────────────────────────────────────
# Run Hyperopt for each model
# ────────────────────────────────────────────────────────────────

best_results = {}

with mlflow.start_run(run_name="Multi_Model_HyperOpt_Comparison"):
    for model_name, space in search_spaces.items():
        print(f"\n=== Tuning {model_name} ===")
        
        trials = Trials()
        
        best = fmin(
            fn=create_objective(
                model_name,
                SVC if model_name == "SVC" else 
                RandomForestClassifier if model_name == "RandomForest" else 
                LogisticRegression,
                space
            ),
            space=space,
            algo=tpe.suggest,
            max_evals=25,               # adjust depending on time budget
            trials=trials,
            verbose=0
        )
        
        # Get best score (remember: we minimized -accuracy)
        best_trial = trials.best_trial
        best_score = -best_trial['result']['loss']
        
        best_results[model_name] = {
            'best_params': best_trial['misc']['vals'],
            'best_cv_accuracy': best_score
        }
        
        print(f"Best CV accuracy: {best_score:.4f}")
        print("Best params:", best_trial['misc']['vals'])

# ────────────────────────────────────────────────────────────────
# Step 4: Select and print the best classifier
# ────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)

best_model_name = None
best_accuracy = -np.inf

for name, res in best_results.items():
    acc = res['best_cv_accuracy']
    print(f"{name:18} → CV Accuracy: {acc:.4f}")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model_name = name

print("\n" + "-"*60)
print(f"WINNER: {best_model_name}")
print(f"Best cross-validation accuracy: {best_accuracy:.4f}")
print("Best parameters:", best_results[best_model_name]['best_params'])
print("-"*60)

# Optional: Log final winner summary
with mlflow.start_run(run_name="Best_Model_Summary"):
    mlflow.log_param("best_model", best_model_name)
    mlflow.log_metric("best_cv_accuracy", best_accuracy)
    mlflow.log_dict(best_results[best_model_name]['best_params'], "best_params.json")