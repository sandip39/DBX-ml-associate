# =============================================================================
# META-OPTIMIZATION: Tuning Method as a Hyperparameter
# Finds which tuning technique (optuna / hyperopt / spark_ml) gives the best clustering
# =============================================================================

%pip install -q sentence-transformers datasets hyperopt optuna kneed mlflow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import mlflow
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
import random

spark = SparkSession.builder.getOrCreate()
mlflow.set_experiment("/Users/your.email@domain.com/AG_News_Meta_Tuning")

# =============================================================================
# COMMON DATA + EMBEDDINGS (done once)
# =============================================================================
print("Loading data and generating embeddings...")

dataset = load_dataset("ag_news", split="train")
df_pd = pd.DataFrame(dataset).sample(n=10000, random_state=42)
df = spark.createDataFrame(df_pd)

# Stratified sampling
strata_col = "label"
total = df.count()
fractions = (df.groupBy(strata_col)
               .agg(F.count("*").alias("cnt"))
               .withColumn("frac", F.col("cnt") / total * 0.10)
               .select(strata_col, "frac")
               .rdd.collectAsMap())

df_sample = df.sampleBy(strata_col, fractions=fractions, seed=42).toPandas()
texts = df_sample["text"].tolist()

model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True, device="cuda")
embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

print(f"Embeddings ready: {embeddings.shape}")

# =============================================================================
# META OBJECTIVE: Optimize over tuning method
# =============================================================================

def meta_objective(trial):
    # Choose tuning method as a categorical hyperparameter
    method = trial.suggest_categorical("tuning_method", ["optuna", "hyperopt", "spark_ml"])
    
    with mlflow.start_run(nested=True, run_name=f"MetaTrial_{trial.number}_{method}") as child_run:
        mlflow.log_param("tuning_method", method)
        
        # ====================== Run selected tuning method ======================
        if method == "optuna":
            import optuna
            def obj(t):
                pca_n = t.suggest_int("pca_n_components", 50, 300, step=25)
                k = t.suggest_int("kmeans_n_clusters", 2, 20)
                init = t.suggest_categorical("init", ["k-means++", "random"])
                scaler = StandardScaler()
                scaled = scaler.fit_transform(embeddings)
                pca = PCA(n_components=pca_n, random_state=42)
                reduced = pca.fit_transform(scaled)
                km = KMeans(n_clusters=k, init=init, random_state=42, n_init=10)
                labels = km.fit_predict(reduced)
                return silhouette_score(reduced, labels)
            
            study = optuna.create_study(direction="maximize")
            study.optimize(obj, n_trials=15)   # fewer inner trials for meta speed
            sil = study.best_value
            best_pca = study.best_params["pca_n_components"]
            best_k = study.best_params["kmeans_n_clusters"]

        elif method == "hyperopt":
            from hyperopt import fmin, tpe, hp, STATUS_OK, SparkTrials
            from hyperopt.pyll import scope
            def obj(params):
                pca_n = int(params["pca_n_components"])
                k = int(params["kmeans_n_clusters"])
                init = params["init"]
                scaler = StandardScaler()
                scaled = scaler.fit_transform(embeddings)
                pca = PCA(n_components=pca_n, random_state=42)
                reduced = pca.fit_transform(scaled)
                km = KMeans(n_clusters=k, init=init, random_state=42, n_init=10)
                labels = km.fit_predict(reduced)
                return {"loss": -silhouette_score(reduced, labels), "status": STATUS_OK}
            
            space = {
                "pca_n_components": scope.int(hp.quniform("pca_n_components", 50, 300, 25)),
                "kmeans_n_clusters": scope.int(hp.quniform("kmeans_n_clusters", 2, 20, 1)),
                "init": hp.choice("init", ["k-means++", "random"])
            }
            trials = SparkTrials(parallelism=4, timeout=600)
            best = fmin(obj, space, tpe.suggest, max_evals=15, trials=trials)
            sil = -trials.best_trial["result"]["loss"]
            best_pca = int(best["pca_n_components"])
            best_k = int(best["kmeans_n_clusters"])

        elif method == "spark_ml":
            from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler, PCA as SparkPCA
            from pyspark.ml.clustering import KMeans as SparkKMeans
            from pyspark.ml import Pipeline
            from pyspark.ml.tuning import ParamGridBuilder
            from pyspark.ml.evaluation import ClusteringEvaluator

            embed_df = spark.createDataFrame(pd.DataFrame(embeddings, columns=[f"f{i}" for i in range(embeddings.shape[1])]))
            assembler = VectorAssembler(inputCols=embed_df.columns, outputCol="features")
            scaler = SparkScaler(inputCol="features", outputCol="scaled")
            pca = SparkPCA(inputCol="scaled", outputCol="pca_features")
            kmeans = SparkKMeans(featuresCol="pca_features", predictionCol="cluster")
            
            param_grid = ParamGridBuilder() \
                .addGrid(pca.k, [50, 100, 150]) \
                .addGrid(kmeans.k, [4, 6, 8, 10, 12]) \
                .build()
            
            evaluator = ClusteringEvaluator(predictionCol="cluster", featuresCol="pca_features", metricName="silhouette")
            
            best_sil = -float('inf')
            for params in param_grid:
                pipeline = Pipeline(stages=[assembler, scaler, pca.copy(params), kmeans.copy(params)])
                fit_model = pipeline.fit(embed_df)
                preds = fit_model.transform(embed_df)
                sil = evaluator.evaluate(preds)
                if sil > best_sil:
                    best_sil = sil
                    best_pca = params[pca.k]
                    best_k = params[kmeans.k]
            sil = best_sil

        # Log results of this meta-trial
        mlflow.log_param("inner_best_pca", best_pca)
        mlflow.log_param("inner_best_k", best_k)
        mlflow.log_metric("silhouette", sil)
        
        return sil   # We want to maximize silhouette

# =============================================================================
# META OPTIMIZATION RUN
# =============================================================================
with mlflow.start_run(run_name="Meta_Optimization_Over_Tuning_Method") as parent_run:
    mlflow.log_param("meta_n_trials", 9)   # Try 9 meta-trials (3 per method roughly)
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(meta_objective, n_trials=9)
    
    print("\n" + "="*60)
    print("META OPTIMIZATION RESULTS")
    print("="*60)
    print(f"Best Tuning Method      : {study.best_params['tuning_method']}")
    print(f"Best Silhouette Score   : {study.best_value:.4f}")
    print(f"Best Meta Trial Number  : {study.best_trial.number}")
    print("="*60)

print("✅ Meta-optimization completed!")
print("Check MLflow UI: You will see one parent run with nested child runs for each meta-trial.")