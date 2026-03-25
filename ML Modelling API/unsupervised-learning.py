# =============================================================================
# AG News Clustering Pipeline - Optimized for Databricks (GPU)
# =============================================================================

# 1. Install required packages (run once)
%pip install -q sentence-transformers kneed datasets mlflow

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
from kneed import KneeLocator

import mlflow
import mlflow.sklearn

# ----------------------------- MLflow Setup -----------------------------
mlflow.set_experiment("/Users/your.email@domain.com/AG_News_Clustering")  # Change to your path

with mlflow.start_run(run_name="gte-large-kmeans-clustering"):
    
    # Log parameters
    mlflow.log_param("dataset", "ag_news")
    mlflow.log_param("sample_size", 10000)
    mlflow.log_param("embedding_model", "Alibaba-NLP/gte-large-en-v1.5")
    mlflow.log_param("pca_components", 100)

    # ----------------------------- 1. LOAD AG NEWS DATA -----------------------------
    print("Loading AG News dataset...")
    dataset = load_dataset("ag_news", split="train")
    df = pd.DataFrame(dataset).sample(n=10000, random_state=42).reset_index(drop=True)

    texts = df["text"].tolist()
    print(f"Loaded {len(texts)} news articles.")

    # ----------------------------- 2. TEXT EMBEDDINGS (GTE-LARGE on GPU) -----------------------------
    print("\nLoading GTE-Large model and generating embeddings on GPU...")
    
    model = SentenceTransformer(
        "Alibaba-NLP/gte-large-en-v1.5",
        trust_remote_code=True,
        device="cuda"   # <-- Key for Databricks GPU acceleration
    )

    embeddings = model.encode(
        texts,
        batch_size=64,           # Increase on better GPUs (A100 → 128+)
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print(f"Embeddings shape: {embeddings.shape}")

    mlflow.log_metric("embedding_dim", embeddings.shape[1])

    # ----------------------------- 3. SCALER + PCA -----------------------------
    print("\nApplying StandardScaler + PCA...")
    scaler = StandardScaler()
    scaled_embeddings = scaler.fit_transform(embeddings)

    pca = PCA(n_components=100, random_state=42)
    reduced_embeddings = pca.fit_transform(scaled_embeddings)

    explained_var = pca.explained_variance_ratio_.sum()
    print(f"Explained variance (100 PCs): {explained_var:.2%}")
    mlflow.log_metric("pca_explained_variance", explained_var)

    # ----------------------------- 4. ELBOW METHOD + KNEELOCATOR -----------------------------
    print("\nElbow method + KneeLocator...")
    inertias = []
    K_range = range(2, 21)

    for k in K_range:
        kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_temp.fit(reduced_embeddings)
        inertias.append(kmeans_temp.inertia_)

    kl = KneeLocator(K_range, inertias, curve="convex", direction="decreasing")
    optimal_k = int(kl.elbow) if kl.elbow is not None else 8
    print(f"Optimal number of clusters: {optimal_k}")

    # Log elbow plot
    plt.figure(figsize=(8,5))
    plt.plot(K_range, inertias, 'o-')
    plt.axvline(optimal_k, color='red', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')
    plt.legend()
    plt.grid(True)
    mlflow.log_figure(plt.gcf(), "elbow_method.png")
    plt.show()

    # ----------------------------- 5. FINAL K-MEANS -----------------------------
    print(f"\nFitting K-Means with k = {optimal_k}...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(reduced_embeddings)

    df["cluster"] = cluster_labels

    # Log model
    mlflow.sklearn.log_model(kmeans, "kmeans_model")

    # ----------------------------- 6. SILHOUETTE SCORE -----------------------------
    sil_score = silhouette_score(reduced_embeddings, cluster_labels)
    print(f"Silhouette Score: {sil_score:.4f}")
    mlflow.log_metric("silhouette_score", sil_score)

    # ----------------------------- 7. VISUALIZATION: 2D PCA + CONVEX HULLS -----------------------------
    print("\nGenerating 2D visualization with convex hulls...")
    pca_2d = PCA(n_components=2, random_state=42)
    embed_2d = pca_2d.fit_transform(reduced_embeddings)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(embed_2d[:, 0], embed_2d[:, 1], 
                          c=cluster_labels, cmap="tab20", alpha=0.7, s=15)

    for c in range(optimal_k):
        points = embed_2d[cluster_labels == c]
        if len(points) >= 3:
            hull = ConvexHull(points)
            hull_pts = points[hull.vertices]
            plt.fill(hull_pts[:, 0], hull_pts[:, 1], alpha=0.2)

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title(f"AG News Clusters (k={optimal_k}) | Silhouette: {sil_score:.4f}")
    plt.legend(*scatter.legend_elements(), title="Clusters", bbox_to_anchor=(1.05, 1))
    plt.grid(True, alpha=0.3)
    
    mlflow.log_figure(plt.gcf(), "cluster_visualization.png")
    plt.show()

    # Optional: Display sample articles per cluster
    print("\nSample articles per cluster:")
    for c in range(optimal_k):
        print(f"\n=== Cluster {c} ({(cluster_labels == c).sum()} articles) ===")
        samples = df[df["cluster"] == c].head(3)["text"]
        for txt in samples:
            print(f"• {txt[:250]}...\n")

    print("\n✅ Pipeline completed! Check MLflow for logged artifacts.")