# ── Impute missing values (only total_bedrooms has NaNs) ───────────────────
imputer = Imputer(
    inputCols=["total_bedrooms"],
    outputCols=["total_bedrooms_imputed"],
    strategy="median"
)

# ── Derived / Interaction features ─────────────────────────────────────────
# We use SQL expressions inside a temporary view or withColumn
df = df.withColumn("rooms_per_household",       col("total_rooms") / col("households"))
df = df.withColumn("bedrooms_per_room",         col("total_bedrooms_imputed") / col("total_rooms"))
df = df.withColumn("population_per_household",  col("population") / col("households"))
df = df.withColumn("income_per_room",           col("median_income") / col("total_rooms"))

# Synthetic listing date (for time-based features)
df = df.withColumn("listing_date", current_date() - (rand() * 365 * 5).cast("integer"))
df = df.withColumn("days_on_market", datediff(current_date(), col("listing_date")))
df = df.withColumn("listing_year", year("listing_date"))

# Simple ordinal encoding for ocean_proximity (alternative to one-hot in some cases)
df = df.withColumn("ocean_score",
    when(col("ocean_proximity") == "NEAR BAY",    4)
   .when(col("ocean_proximity") == "NEAR OCEAN",  3)
   .when(col("ocean_proximity") == "<1H OCEAN",   2)
   .when(col("ocean_proximity") == "ISLAND",      5)
   .otherwise(1)
)

# ── Numerical columns we want to scale/assemble ────────────────────────────
numerical_features = [
    "housing_median_age", "total_rooms", "total_bedrooms_imputed",
    "population", "households", "median_income",
    "rooms_per_household", "bedrooms_per_room", "population_per_household",
    "income_per_room", "days_on_market", "listing_year", "ocean_score"
]

# ── StringIndexer + OneHotEncoder for categorical feature ──────────────────
indexer = StringIndexer(
    inputCol="ocean_proximity",
    outputCol="ocean_proximity_indexed",
    handleInvalid="keep"
)

encoder = OneHotEncoder(
    inputCols=["ocean_proximity_indexed"],
    outputCols=["ocean_proximity_onehot"],
    dropLast=True   # avoid dummy variable trap
)



# ── Assemble numerical features ────────────────────────────────────────────
assembler_num = VectorAssembler(
    inputCols=numerical_features,
    outputCol="num_features_raw",
    handleInvalid="skip"
)

# ── Scale numerical features ───────────────────────────────────────────────
scaler = StandardScaler(
    inputCol="num_features_raw",
    outputCol="num_features_scaled",
    withMean=True,
    withStd=True
)

# ── Final assembler: combine scaled numerics + one-hot ─────────────────────
final_assembler = VectorAssembler(
    inputCols=["num_features_scaled", "ocean_proximity_onehot"],
    outputCol="features",
    handleInvalid="skip"
)

# ── Full pipeline ──────────────────────────────────────────────────────────
pipeline = Pipeline(stages=[
    imputer,                    # 1. Fill missing values
    indexer,                    # 2. String → numeric index
    encoder,                    # 3. One-hot encoding
    assembler_num,              # 4. Combine raw numerics
    scaler,                     # 5. Standardize numerics
    final_assembler             # 6. Final feature vector
])

# Fit the pipeline (in real ML: fit only on train set!)
pipeline_model = pipeline.fit(df)

# Transform the data
df_prepared = pipeline_model.transform(df)

# ── Inspect result ─────────────────────────────────────────────────────────
print("Final feature vector preview:")
df_prepared.select(
    "median_house_value",           # target
    "features"                      # assembled & scaled vector
).show(5, truncate=False)

# Optional: see vector size
from pyspark.ml.functions import vector_to_array
df_prepared.selectExpr("size(vector_to_array(features)) as feature_count").show(1)