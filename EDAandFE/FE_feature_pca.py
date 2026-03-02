# =====================================================================
# PySpark Pipeline: Imputation + OHE + Scaling + PCA
# California Housing Dataset – Databricks Notebook
# =====================================================================

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler,
    PCA
)
from pyspark.ml.functions import vector_to_array

spark = SparkSession.builder.appName("HousingImputeOHE_PCA").getOrCreate()

# ── 1. Load the CSV file ───────────────────────────────────────────────────
# Public URL – works directly in Databricks
csv_url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

df = (spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(csv_url))

df.cache()
print("Original row count:", df.count())
df.printSchema()

# ── Imputation (median strategy – standard / regular approach) ─────────────
imputer = Imputer(
    inputCols=["total_bedrooms"],
    outputCols=["total_bedrooms_imputed"],
    strategy="median"
)

# ── One-Hot Encoding for categorical feature ───────────────────────────────
# StringIndexer first → then OneHotEncoder
indexer = StringIndexer(
    inputCol="ocean_proximity",
    outputCol="ocean_proximity_index",
    handleInvalid="keep"          # safe for production
)

ohe = OneHotEncoder(
    inputCols=["ocean_proximity_index"],
    outputCols=["ocean_proximity_ohe"],
    dropLast=True                 # avoid dummy variable trap
)

# ── Numerical columns to be assembled & scaled ─────────────────────────────
numerical_cols = [
    "housing_median_age",
    "total_rooms",
    "total_bedrooms_imputed",
    "population",
    "households",
    "median_income",
    "latitude",
    "longitude"
]

# ── Assemble numerical features ────────────────────────────────────────────
num_assembler = VectorAssembler(
    inputCols=numerical_cols,
    outputCol="num_features_raw",
    handleInvalid="skip"
)

# ── Standard scaling (strongly recommended before PCA) ─────────────────────
scaler = StandardScaler(
    inputCol="num_features_raw",
    outputCol="num_features_scaled",
    withMean=True,
    withStd=True
)

# ── PCA – reduce dimensionality ────────────────────────────────────────────
pca = PCA(
    k=6,                           # ← tune this: try 4–10
    inputCol="num_features_scaled",
    outputCol="pca_features"
)

# ── Optional: final assembler if you want to combine PCA + OHE ─────────────
final_assembler = VectorAssembler(
    inputCols=["pca_features", "ocean_proximity_ohe"],
    outputCol="final_features",
    handleInvalid="skip"
)

# ── Build the complete pipeline ────────────────────────────────────────────
pipeline = Pipeline(stages=[
    imputer,
    indexer,
    ohe,
    num_assembler,
    scaler,
    pca,
    final_assembler             # combine PCA + categorical
])

# Fit the pipeline
pipeline_model = pipeline.fit(df)

# Transform the data
df_transformed = pipeline_model.transform(df)

# ── Inspect results ────────────────────────────────────────────────────────
print("\n=== After Pipeline (first 5 rows) ===")
df_transformed.select(
    "median_house_value",
    "pca_features",
    "ocean_proximity_ohe",
    "final_features"
).show(5, truncate=False)

# Show PCA component count
df_transformed.selectExpr("size(pca_features) as pca_dim").show(1)

# ── Explained Variance (very important for choosing k) ─────────────────────
pca_stage = pipeline_model.stages[-3]  # PCA is the 3rd from end
explained_var = pca_stage.explainedVariance

print("\nExplained Variance per Principal Component:")
cum_var = 0.0
for i, var in enumerate(explained_var):
    cum_var += var
    print(f"PC{i+1}: {var:.4f}  (cumulative: {cum_var:.4f} / {cum_var*100:.1f}%)")

# Example: find minimal k for ≥ 90% variance
cum = 0.0
k_needed = 0
for i, v in enumerate(explained_var, 1):
    cum += v
    if cum >= 0.90:
        k_needed = i
        break

print(f"\nMinimum components needed for ≥90% variance: {k_needed}")


