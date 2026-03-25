# =============================================================================
# Proportionate Stratified Sampling in Databricks
# Keeps the exact same category proportions as the original data
# =============================================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, sum as spark_sum
import pyspark.sql.functions as F

spark = SparkSession.builder.getOrCreate()

# ----------------------------- Load your DataFrame -----------------------------
# Replace this with your actual data source
df = spark.table("your_catalog.db.ag_news")          # or spark.read.format("delta").load("/path")

# Or if using the AG News dataset from previous code:
# df = spark.createDataFrame(pd.DataFrame(dataset))   # if converting from pandas

strata_col = "label"          # Change to your stratification column (e.g., "category", "label", "topic")

# ----------------------------- 1. Calculate Proportions Automatically -----------------------------
# Get count and proportion for each category
total_count = df.count()

proportions_df = (
    df.groupBy(strata_col)
      .agg(count("*").alias("count"))
      .withColumn("proportion", F.col("count") / total_count)
)

# Convert to Python dict for sampleBy()
fractions = {row[strata_col]: float(row["proportion"]) for row in proportions_df.collect()}

print("Sampling fractions (same as original proportions):")
for k, v in fractions.items():
    print(f"  {k}: {v:.4f} ({v*100:.2f}%)")

# ----------------------------- 2. Perform Stratified Sampling -----------------------------
sample_fraction = 0.10   # Change this: 0.05 = 5%, 0.20 = 20%, etc.

stratified_sample = df.sampleBy(
    col=strata_col,
    fractions={k: v * sample_fraction for k, v in fractions.items()},
    seed=42
)

# ----------------------------- 3. Verify the Results -----------------------------
original_dist = df.groupBy(strata_col).count().orderBy(strata_col)
sample_dist   = stratified_sample.groupBy(strata_col).count().orderBy(strata_col)

print(f"\nOriginal total rows     : {df.count():,}")
print(f"Stratified sample rows  : {stratified_sample.count():,}")
print(f"Actual sampling rate    : {stratified_sample.count() / df.count():.4f}")

print("\n=== Original Distribution ===")
original_dist.show()

print("\n=== Sample Distribution (should be very close) ===")
sample_dist.show()

# ----------------------------- 4. Save the Sample (Optional) -----------------------------
stratified_sample.write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("your_catalog.db.ag_news_stratified_sample_10pct")