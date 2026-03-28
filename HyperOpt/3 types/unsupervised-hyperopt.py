# =============================================================================
# CODE SET 1/3: Hyperopt + SparkTrials – Parent + Nested Child Runs
# =============================================================================

%pip install -q sentence-transformers datasets hyperopt kneed mlflow

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK, SparkTrials
from hyperopt.pyll import scope
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
mlflow.set_experiment("/Users/your.email@domain.com/AG_News_Hyperopt")

# ====================== PARENT RUN ======================
with mlflow.start_run(run_name="Hyperopt_SparkTrials_Parent") as parent_run:
    mlflow.log_param("method", "hyperopt_sparktrials")
    mlflow.log_param("embedding_model", "Alibaba-NLP/gte-large-en-v1.5")
    mlflow.log_param("sample_size", 10000)

    # 1. Load + Stratified Sampling
    dataset = load_dataset("ag_news", split="train")
    df_pd = pd.DataFrame(dataset).sample(n=10000, random_state=42)
    df = spark.createDataFrame(df_pd)
    
    strata_col = "label"
    total = df.count()
    fractions = (df.groupBy(strata_col)
                   .agg(F.count("*").alias("cnt"))
                   .withColumn("frac", F.col("cnt") / total * 0.10)
                   .select(strata_col, "frac")
                   .rdd.collectAsMap())
    
    df_sample = df.sampleBy(strata_col, fractions=fractions, seed=42).toPandas()
    texts = df_sample["text"].tolist()

    # 2. Embeddings
    model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True, device="cuda")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    # 3. Objective with nested child run
    def objective(params):
        pca_n = int(params["pca_n_components"])
        k = int(params["kmeans_n_clusters"])
        init_mode = params["init"]
        
        with mlflow.start_run(nested=True, run_name=f"Trial_PCA{pca_n}_K{k}") as child_run:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(embeddings)
            pca = PCA(n_components=pca_n, random_state=42)
            reduced = pca.fit_transform(scaled)
            
            km = KMeans(n_clusters=k, init=init_mode, random_state=42, n_init=10)
            labels = km.fit_predict(reduced)
            sil = silhouette_score(reduced, labels)
            
            mlflow.log_params({
                "pca_n_components": pca_n,
                "kmeans_n_clusters": k,
                "init": init_mode
            })
            mlflow.log_metric("silhouette", sil)
        
        return {"loss": -sil, "status": STATUS_OK}

    search_space = {
        "pca_n_components": scope.int(hp.quniform("pca_n_components", 50, 300, 25)),
        "kmeans_n_clusters": scope.int(hp.quniform("kmeans_n_clusters", 2, 20, 1)),
        "init": hp.choice("init", ["k-means++", "random"])
    }

    spark_trials = SparkTrials(parallelism=6, timeout=1800)

    best = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=30,
        trials=spark_trials,
        rstate=np.random.default_rng(42)
    )

    # Log best to parent
    best_sil = -spark_trials.best_trial["result"]["loss"]
    mlflow.log_param("best_pca_components", int(best["pca_n_components"]))
    mlflow.log_param("best_k", int(best["kmeans_n_clusters"]))
    mlflow.log_metric("best_silhouette", best_sil)

    print("✅ Hyperopt Parent + Nested Child Runs completed.")