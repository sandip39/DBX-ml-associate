from pyspark.sql import functions as F

# 1. Load your housing data
df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/path/to/housing_data.csv")

# 2. Identify all numeric columns
numeric_cols = [f.name for f in df.schema.fields if str(f.dataType) in ["DoubleType", "IntegerType", "FloatType", "LongType"]]

# 3. Calculate percentiles (e.g., 1st and 99th) for all numeric columns
# approxQuantile returns a list of values for each column
lower_quantile = 0.01
upper_quantile = 0.99
thresholds = {}

for col in numeric_cols:
    # Get [1st percentile, 99th percentile]
    thresholds[col] = df.stat.approxQuantile(col, [lower_quantile, upper_quantile], 0.01)

# 4. Apply filtering to remove outliers across all numeric columns
df_cleaned = df
for col in numeric_cols:
    lower_bound = thresholds[col][0]
    upper_bound = thresholds[col][1]
    
    # Filter rows within the bounds
    df_cleaned = df_cleaned.filter((F.col(col) >= lower_bound) & (F.col(col) <= upper_bound))

display(df_cleaned)
