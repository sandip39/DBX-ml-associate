import pandas as pd

# Load directly into pandas (fast for ~20k rows)
df = pd.read_csv(url)

print("Original shape:", df.shape)
print("Categorical columns:", df.select_dtypes(include='object').columns.tolist())
print("\nocean_proximity value counts:\n", df['ocean_proximity'].value_counts())

# ── One-hot encode ocean_proximity ────────────────────────────────────────
# drop_first=True avoids multicollinearity (dummy variable trap) – good for linear models
df_encoded = pd.get_dummies(
    df,
    columns=['ocean_proximity'],
    prefix='ocean',
    drop_first=True,           # drops one category (usually the most frequent or first)
    dtype=int                  # use 0/1 instead of True/False
)

# Alternative: keep all categories (useful for trees)
# df_encoded = pd.get_dummies(df, columns=['ocean_proximity'], prefix='ocean', dtype=int)

print("\nShape after one-hot encoding:", df_encoded.shape)
print("New columns:", [c for c in df_encoded.columns if c.startswith('ocean_')])

# Preview
display(df_encoded.filter(regex='ocean_proximity|ocean_').head(8))

#spark version

from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("HousingOneHot").getOrCreate()

# Load data
df = (spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(url))

df.cache()
df.printSchema()

# ── Step 1: Convert string category → numeric index ────────────────────────
indexer = StringIndexer(
    inputCol="ocean_proximity",
    outputCol="ocean_proximity_indexed",
    handleInvalid="keep"   # useful if new categories appear later
)

# ── Step 2: One-hot encode the indexed column ──────────────────────────────
encoder = OneHotEncoder(
    inputCols=["ocean_proximity_indexed"],
    outputCols=["ocean_proximity_onehot"],
    dropLast=True          # same as drop_first – avoids multicollinearity
    # dropLast=False       # keep all categories if preferred
)

# ── Build & run pipeline ───────────────────────────────────────────────────
pipeline = Pipeline(stages=[indexer, encoder])
model = pipeline.fit(df)
df_encoded_spark = model.transform(df)

# Show result
print("\nAfter one-hot encoding:")
df_encoded_spark.select(
    "ocean_proximity",
    "ocean_proximity_indexed",
    "ocean_proximity_onehot"
).show(10, truncate=False)

# Optional: convert sparse vector to dense columns (for easier viewing / pandas)
from pyspark.ml.functions import vector_to_array

df_with_dense = df_encoded_spark.withColumn(
    "ocean_onehot_array", vector_to_array("ocean_proximity_onehot")
)

# If you want separate columns (less common in Spark ML)
# You can use OneHotEncoder + VectorAssembler later in full pipeline
display(df_with_dense.select("ocean_proximity", "ocean_onehot_array").limit(10))


