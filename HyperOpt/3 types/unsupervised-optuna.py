# =============================================================================
# CODE SET 3/3: Optuna – Explicit Parent + nested=True Child Runs
# =============================================================================

%pip install -q sentence-transformers datasets optuna mlflow

import pandas as pd
import mlflow
import optuna
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

spark = SparkSession.builder.getOrCreate()
mlflow.set_experiment("/Users/your.email@domain.com/AG_News_Optuna")

def objective(trial):
    with mlflow.start_run(nested=True, run_name=f"Optuna_Trial_{trial.number}") as child_run:
        pca_n = trial.suggest_int("pca_n_components", 50, 300, step=25)
        k = trial.suggest_int("kmeans_n_clusters", 2, 20)
        init_mode = trial.suggest_categorical("init", ["k-means++", "random"])
        
        scaler = StandardScaler()
        scaled = scaler.fit_transform(embeddings)
        pca = PCA(n_components=pca_n, random_state=42)
        reduced = pca.fit_transform(scaled)
        
        km = KMeans(n_clusters=k, init=init_mode, random_state=42, n_init=10)
        labels = km.fit_predict(reduced)
        sil = silhouette_score(reduced, labels)
        
        mlflow.log_params(trial.params)
        mlflow.log_metric("silhouette", sil)
        
        return sil

# ====================== PARENT RUN ======================
with mlflow.start_run(run_name="Optuna_AGNews_Parent") as parent_run:
    # Load + Stratified Sampling + Embeddings
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

    model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True, device="cuda")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30)
    
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_silhouette", study.best_value)

    print("✅ Optuna Parent + Nested Child Runs completed.")