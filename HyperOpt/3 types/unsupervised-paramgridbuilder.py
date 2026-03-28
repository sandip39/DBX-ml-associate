# =============================================================================
# CODE SET 2/3: Spark ML ParamGridBuilder – Manual Parent + Nested Child Runs
# =============================================================================

%pip install -q sentence-transformers datasets mlflow

import pandas as pd
import mlflow
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler as SparkScaler, PCA as SparkPCA
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import ClusteringEvaluator
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

spark = SparkSession.builder.getOrCreate()
mlflow.set_experiment("/Users/your.email@domain.com/AG_News_SparkML_Manual")

# ====================== PARENT RUN ======================
with mlflow.start_run(run_name="SparkML_ParamGrid_Parent") as parent_run:
    mlflow.log_param("method", "spark_ml_paramgrid")

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
    
    embed_df = spark.createDataFrame(pd.DataFrame(embeddings, columns=[f"f{i}" for i in range(embeddings.shape[1])]))

    # 3. Manual nested runs for each parameter combination (simulated)
    # Note: Since Spark ML tuning is not easily loopable, we manually loop over param grid for nested runs
    assembler = VectorAssembler(inputCols=embed_df.columns, outputCol="features")
    scaler = SparkScaler(inputCol="features", outputCol="scaled")
    pca = SparkPCA(inputCol="scaled", outputCol="pca_features")
    kmeans = SparkKMeans(featuresCol="pca_features", predictionCol="cluster")

    param_grid = ParamGridBuilder() \
        .addGrid(pca.k, [50, 100, 150, 200]) \
        .addGrid(kmeans.k, [4, 6, 8, 10, 12, 15]) \
        .addGrid(kmeans.initMode, ["k-means||", "random"]) \
        .build()

    evaluator = ClusteringEvaluator(predictionCol="cluster", featuresCol="pca_features", metricName="silhouette")

    best_sil = -float('inf')
    best_params = None

    for i, params in enumerate(param_grid):
        with mlflow.start_run(nested=True, run_name=f"ParamCombo_{i}") as child_run:
            # Build pipeline with current params
            current_pipeline = Pipeline(stages=[assembler, scaler, pca.copy(params), kmeans.copy(params)])
            model_fit = current_pipeline.fit(embed_df)
            predictions = model_fit.transform(embed_df)
            sil = evaluator.evaluate(predictions)
            
            mlflow.log_params(params)
            mlflow.log_metric("silhouette", sil)
            
            if sil > best_sil:
                best_sil = sil
                best_params = params

    mlflow.log_metric("best_silhouette", best_sil)
    print("✅ Spark ML Parent + Manual Nested Child Runs completed.")