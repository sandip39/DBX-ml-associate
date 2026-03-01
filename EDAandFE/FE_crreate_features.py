# =============================================
# FEATURE ENGINEERING EXAMPLES IN DATABRICKS
# PySpark code for a new Databricks Notebook
# Uses standard California Housing dataset
# (loaded directly from public CSV – works perfectly in Databricks)
# =============================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.feature import *
from pyspark.ml import Pipeline
import pyspark.sql.types as T

spark = SparkSession.builder.appName("HousingFeatureEngineering").getOrCreate()

# --------------------- 1. LOAD DATA FROM CSV ---------------------
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

df = (spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(url))

df.cache()
print("Original schema:")
df.printSchema()
df.show(5, truncate=False)


# Interaction Features - (Products, ratios, or combinations of existing columns)


df = (df
      .withColumn("rooms_per_household", col("total_rooms") / col("households"))
      .withColumn("bedrooms_per_room", col("total_bedrooms") / col("total_rooms"))
      .withColumn("population_per_household", col("population") / col("households"))
      .withColumn("income_age_interaction", col("median_income") * col("housing_median_age"))   # true interaction term
      .withColumn("income_per_room", col("median_income") / col("total_rooms")))

df.select(
    "total_rooms", "households", "median_income", "housing_median_age",
    "rooms_per_household", "bedrooms_per_room", "income_age_interaction"
).show(5)

#Binning (Discretization) - Manual when() + MLlib Bucketizer

# Manual binning (housing_median_age)
df = df.withColumn("age_category",
    when(col("housing_median_age") < 15, "Very New")
    .when(col("housing_median_age") < 30, "New")
    .when(col("housing_median_age") < 45, "Mid-Age")
    .otherwise("Old")
)

# Bucketizer on median_income (data-driven splits)
splits = [0.0, 1.5, 3.0, 4.5, 6.0, float("inf")]
bucketizer = Bucketizer(splits=splits, inputCol="median_income", outputCol="income_bucket")

df = bucketizer.transform(df)

df.select("housing_median_age", "age_category", "median_income", "income_bucket").show(5)


#Time-based Features - (Original data has no date → we add a realistic synthetic listing_date)

# Add synthetic listing_date (last 0–5 years)
df = df.withColumn("listing_date", 
                   date_add(current_date(), -(rand() * 365 * 5).cast("integer")))

# Extract useful time features
df = (df
      .withColumn("listing_year", year("listing_date"))
      .withColumn("listing_month", month("listing_date"))
      .withColumn("listing_dayofweek", dayofweek("listing_date"))
      .withColumn("is_weekend", when(col("listing_dayofweek").isin([1, 7]), 1).otherwise(0))
      .withColumn("days_on_market", datediff(current_date(), col("listing_date")))
      .withColumn("season", 
                  when(col("listing_month").isin([12,1,2]), "Winter")
                  .when(col("listing_month").isin([3,4,5]), "Spring")
                  .when(col("listing_month").isin([6,7,8]), "Summer")
                  .otherwise("Fall")))

df.select("listing_date", "listing_year", "is_weekend", "days_on_market", "season").show(5)


#CountVectorizer (Text / NLP Features)
#Housing dataset has no text column, so here is a clean self-contained example you can run right after the housing load (or join later).

# Sample real-estate description data
text_data = [
    (1, "Spacious 3 bedroom modern house with ocean view and large backyard near beach California"),
    (2, "Cozy family home in quiet neighborhood with excellent schools and park walking distance"),
    (3, "Luxury villa with swimming pool garage and stunning mountain views in exclusive gated community"),
    (4, "Affordable starter home close to downtown public transport and shopping centers")
]

schema = T.StructType([
    T.StructField("house_id", T.IntegerType()),
    T.StructField("description", T.StringType())
])

text_df = spark.createDataFrame(text_data, schema)

# Pipeline: Tokenizer → CountVectorizer
tokenizer = Tokenizer(inputCol="description", outputCol="words")
count_vec = CountVectorizer(inputCol="words", outputCol="features", vocabSize=30, minDF=1)

pipeline = Pipeline(stages=[tokenizer, count_vec])
cv_model = pipeline.fit(text_df)
result = cv_model.transform(text_df)

result.select("description", "features").show(truncate=False)

# See the learned vocabulary
print("Vocabulary:", cv_model.stages[-1].vocabulary)

# Domain-Specific Features
#(Real-world housing knowledge – ratios, flags, derived metrics)

df = (df
      .withColumn("bedrooms_per_room", col("total_bedrooms") / col("total_rooms"))           # very common in housing models
      .withColumn("rooms_per_person", col("total_rooms") / col("population"))
      .withColumn("high_density_area", when(col("population_per_household") > 4.5, 1).otherwise(0))
      .withColumn("ocean_proximity_score",                                     # domain ordering
                  when(col("ocean_proximity") == "NEAR BAY", 4)
                  .when(col("ocean_proximity") == "NEAR OCEAN", 3)
                  .when(col("ocean_proximity") == "<1H OCEAN", 2)
                  .when(col("ocean_proximity") == "ISLAND", 5)
                  .otherwise(1))  # INLAND = 1
      .withColumn("affordability_index", col("median_income") / (col("median_house_value") / 100000))  # income relative to price
      .withColumn("price_category", 
                  when(col("median_house_value") >= 450000, "Luxury")
                  .when(col("median_house_value") >= 250000, "Mid-range")
                  .otherwise("Affordable")))

df.select(
    "ocean_proximity", "ocean_proximity_score",
    "median_income", "median_house_value", "affordability_index",
    "price_category", "high_density_area"
).show(5)

from pyspark.ml.feature import VectorAssembler

feature_cols = [
    "rooms_per_household", "bedrooms_per_room", "population_per_household",
    "income_age_interaction", "income_per_room", "income_bucket",
    "listing_year", "days_on_market", "is_weekend",
    "bedrooms_per_room", "rooms_per_person", "high_density_area",
    "ocean_proximity_score", "affordability_index"
]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
final_df = assembler.transform(df)

final_df.select("features", "median_house_value").show(3, truncate=False)


# Modifying above for Pandas

# =============================================
# FEATURE ENGINEERING EXAMPLES – PANDAS VERSION
# Starts with Spark → converts to pandas early
# Uses California Housing dataset from public CSV
# =============================================

from pyspark.sql import SparkSession
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

spark = SparkSession.builder.appName("HousingPandasFeatures").getOrCreate()

# ── 1. LOAD DATA FROM CSV (Spark) ────────────────────────────────────────
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

spark_df = (spark.read
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(url))

# ── Convert to pandas DataFrame early ─────────────────────────────────────
df = spark_df.toPandas()

print("Pandas DataFrame shape:", df.shape)
print(df.head())
print("\nColumns:", df.columns.tolist())

# Create interaction / ratio features
df['rooms_per_household']       = df['total_rooms'] / df['households']
df['bedrooms_per_room']         = df['total_bedrooms'] / df['total_rooms']
df['population_per_household']  = df['population'] / df['households']
df['income_age_interaction']    = df['median_income'] * df['housing_median_age']
df['income_per_room']           = df['median_income'] / df['total_rooms']

print(df[['total_rooms', 'households', 'median_income', 
          'rooms_per_household', 'bedrooms_per_room', 'income_age_interaction']].head())



# Manual binning – housing_median_age
bins_age = [0, 15, 30, 45, np.inf]
labels_age = ['Very New', 'New', 'Mid-Age', 'Old']
df['age_category'] = pd.cut(df['housing_median_age'], bins=bins_age, labels=labels_age, right=False)

# Quantile-based or fixed binning on median_income
df['income_bucket'] = pd.qcut(df['median_income'], q=5, labels=[0,1,2,3,4], duplicates='drop')
# or fixed bins:
# df['income_bucket'] = pd.cut(df['median_income'], bins=[0, 1.5, 3, 4.5, 6, np.inf], labels=[0,1,2,3,4])

print(df[['housing_median_age', 'age_category', 'median_income', 'income_bucket']].head())


# Synthetic listing date: random within last ~5 years
np.random.seed(42)  # for reproducibility
days_back = np.random.randint(0, 365*5, size=len(df))
df['listing_date'] = pd.to_datetime('today') - pd.to_timedelta(days_back, unit='D')

# Extract features
df['listing_year']     = df['listing_date'].dt.year
df['listing_month']    = df['listing_date'].dt.month
df['listing_dayofweek']= df['listing_date'].dt.dayofweek          # 0 = Monday, 6 = Sunday
df['is_weekend']       = df['listing_dayofweek'].isin([5,6]).astype(int)
df['days_on_market']   = (pd.to_datetime('today') - df['listing_date']).dt.days

# Simple season
def get_season(month):
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Fall'

df['season'] = df['listing_month'].apply(get_season)

print(df[['listing_date', 'listing_year', 'is_weekend', 'days_on_market', 'season']].head())


# Create sample text data (real-estate descriptions)
text_samples = [
    "Spacious 3 bedroom modern house with ocean view and large backyard near beach California",
    "Cozy family home in quiet neighborhood with excellent schools and park walking distance",
    "Luxury villa with swimming pool garage and stunning mountain views in exclusive gated community",
    "Affordable starter home close to downtown public transport and shopping centers",
    "Renovated condo downtown with rooftop terrace and city skyline view"
]

# For demo: assign random descriptions to housing rows (or use real ones if available)
np.random.seed(42)
df['description'] = np.random.choice(text_samples, size=len(df), replace=True)

# Apply CountVectorizer
vectorizer = CountVectorizer(max_features=30, stop_words='english', min_df=5)
X_text = vectorizer.fit_transform(df['description'])

# Convert to DataFrame with feature names
text_features_df = pd.DataFrame(
    X_text.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=df.index
)

# Merge back (or keep separate for sparse handling)
df = pd.concat([df, text_features_df.add_prefix('cv_')], axis=1)

print("Vocabulary size:", len(vectorizer.get_feature_names_out()))
print(df.filter(like='cv_').head(3))

# Common housing domain features
df['rooms_per_person']       = df['total_rooms'] / df['population']
df['high_density_area']      = (df['population_per_household'] > 4.5).astype(int)

# Ordinal encoding for ocean_proximity (domain knowledge)
ocean_map = {
    'NEAR BAY':    4,
    'NEAR OCEAN':  3,
    '<1H OCEAN':   2,
    'ISLAND':      5,
    'INLAND':      1
}
df['ocean_proximity_score'] = df['ocean_proximity'].map(ocean_map).fillna(1)

# Affordability rough index
df['affordability_index'] = df['median_income'] / (df['median_house_value'] / 100000)

# Price category
price_bins = [0, 250000, 450000, np.inf]
price_labels = ['Affordable', 'Mid-range', 'Luxury']
df['price_category'] = pd.cut(df['median_house_value'], bins=price_bins, labels=price_labels, right=False)

print(df[['ocean_proximity', 'ocean_proximity_score', 'affordability_index', 
          'price_category', 'high_density_area']].head())


# Example feature list (numeric + encoded)
feature_cols = [
    'rooms_per_household', 'bedrooms_per_room', 'population_per_household',
    'income_age_interaction', 'income_per_room',
    'listing_year', 'days_on_market', 'is_weekend',
    'rooms_per_person', 'high_density_area', 'ocean_proximity_score',
    'affordability_index'
    # + text_features_df columns if desired
]

X = df[feature_cols].copy()
y = df['median_house_value']

print("Ready for modeling – X shape:", X.shape)
print(X.head(3))


