# =============================================
# FEATURE SELECTION – California Housing Dataset
# Pandas + scikit-learn – Databricks Notebook
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest, f_regression, mutual_info_regression,
    SelectFromModel
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error

# ── 1. Load data ───────────────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
df = pd.read_csv(url)

print("Shape:", df.shape)
print("\nMissing values:\n", df.isna().sum())

# Quick imputation (median for total_bedrooms – only column with NaNs)
df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())


# Features & target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Train/test split (important for honest feature selection)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Preprocessing pipeline ─────────────────────────────────────────────────
num_features = X_train.select_dtypes(include=['float64', 'int64']).columns
cat_features = X_train.select_dtypes(include=['object']).columns  # → 'ocean_proximity'

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), cat_features)
    ])

# Fit & transform
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep  = preprocessor.transform(X_test)

# Get feature names after transformation
cat_ohe_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
all_feature_names = np.concatenate([num_features, cat_ohe_names])

print("Features after preprocessing:", len(all_feature_names))
print(all_feature_names)

#A. Correlation-based feature selection

# Convert to DataFrame for correlation
X_train_df = pd.DataFrame(X_train_prep, columns=all_feature_names)

# Correlation with target (absolute value)
correlations = X_train_df.corrwith(y_train).abs().sort_values(ascending=False)

print("\nTop correlations with target:")
print(correlations.head(12))

# Visualize
plt.figure(figsize=(10, 6))
correlations.sort_values().plot(kind='barh', color='skyblue')
plt.title("Absolute Correlation with Median House Value")
plt.show()

# Example: keep features with corr > 0.3
strong_corr_features = correlations[correlations > 0.3].index.tolist()
print("\nFeatures with |corr| > 0.3:", strong_corr_features)

# B. Mutual Information-based feature selection
mi_scores = mutual_info_regression(X_train_prep, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=all_feature_names).sort_values(ascending=False)

print("\nMutual Information scores:")
print(mi_series.head(12))

# Plot
plt.figure(figsize=(10, 6))
mi_series.sort_values().tail(15).plot(kind='barh', color='teal')
plt.title("Mutual Information – Feature Importance")
plt.show()


# C. Recursive Feature Elimination (RFE)

# Use simple linear model for RFE
model_rfe = LinearRegression()
rfe = RFE(estimator=model_rfe, n_features_to_select=8)

rfe.fit(X_train_prep, y_train)

selected_rfe = all_feature_names[rfe.support_]
ranking_rfe = pd.Series(rfe.ranking_, index=all_feature_names).sort_values()

print("\nRFE Selected features (top 8):")
print(selected_rfe.tolist())

print("\nFeature ranking (1 = selected):")
print(ranking_rfe)

# D. Feature Importance from Trees
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_prep, y_train)

importances = pd.Series(rf.feature_importances_, index=all_feature_names).sort_values(ascending=False)

print("\nRandom Forest Feature Importances:")
print(importances.head(12))

# Plot
plt.figure(figsize=(10, 6))
importances.head(15).plot(kind='bar', color='coral')
plt.title("Random Forest Feature Importance")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Optional: SelectFromModel (threshold-based)
selector_rf = SelectFromModel(rf, threshold="median", prefit=True)
selected_rf = all_feature_names[selector_rf.get_support()]
print("\nFeatures selected by RF (above median importance):", selected_rf.tolist())

# E. K best features
kbest = SelectKBest(score_func=f_regression, k=10)
X_train_kbest = kbest.fit_transform(X_train_prep, y_train)

selected_features_kbest = all_feature_names[kbest.get_support()]
scores_kbest = pd.Series(kbest.scores_, index=all_feature_names).sort_values(ascending=False)

print("\nTop 10 features – SelectKBest (f_regression):")
print(scores_kbest.head(10))

print("\nSelected features:", selected_features_kbest.tolist())


top_n = 10

summary = pd.DataFrame({
    'Correlation': correlations.head(top_n).index,
    'Mutual Info': mi_series.head(top_n).index,
    'f_regression': scores_kbest.head(top_n).index,
    'RandomForest': importances.head(top_n).index
})

print("\nTop features by different methods:")
display(summary)


# spark


# =====================================================================
# FEATURE SELECTION – California Housing Dataset (SPARK VERSION)
# PySpark MLlib – Databricks Notebook
# =====================================================================

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import (
    VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder,
    Imputer, ChiSqSelector, VarianceThresholdSelector
)
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.stat import Correlation
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import DoubleType

spark = SparkSession.builder.appName("HousingFeatureSelectionSpark").getOrCreate()

# ── 1. Load data ───────────────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"

df = (spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(url))

df = df.cache()
df.printSchema()

# Impute missing values
imputer = Imputer(inputCols=["total_bedrooms"], outputCols=["total_bedrooms"], strategy="median")

# String index + One-hot encode categorical feature
indexer = StringIndexer(inputCol="ocean_proximity", outputCol="ocean_idx", handleInvalid="keep")
encoder = OneHotEncoder(inputCols=["ocean_idx"], outputCols=["ocean_onehot"], dropLast=True)

# Numerical features (after imputation)
num_cols = [
    "housing_median_age", "total_rooms", "total_bedrooms",
    "population", "households", "median_income",
    "latitude", "longitude"
]

# Assemble numerical features
assembler_num = VectorAssembler(inputCols=num_cols, outputCol="num_features_raw")

# Scale numerical features
scaler = StandardScaler(inputCol="num_features_raw", outputCol="num_features_scaled", withMean=True, withStd=True)

# Final assembler (numerical scaled + one-hot)
final_assembler = VectorAssembler(
    inputCols=["num_features_scaled", "ocean_onehot"],
    outputCol="features_raw"
)

preprocess_pipeline = Pipeline(stages=[
    imputer, indexer, encoder, assembler_num, scaler, final_assembler
])

preprocess_model = preprocess_pipeline.fit(df)
df_prepared = preprocess_model.transform(df)

df_prepared.select("features_raw", "median_house_value").show(3, truncate=False)


# Compute correlation matrix (Spark way)
corr_matrix = Correlation.corr(
    df_prepared.select("features_raw", "median_house_value"),
    "features_raw"
).head()[0].toArray()

# Get feature names in order
feature_names = num_cols + [f"ocean_onehot_{i}" for i in range(df_prepared.select("ocean_onehot").head()[0].size)]

# Correlation of each feature with target (assuming target is last in vector for simplicity)
# → Better: assemble target too or use pandas conversion for small data
df_pandas_corr = df_prepared.select("features_raw", "median_house_value").toPandas()
X_pd = pd.DataFrame([v.toArray() for v in df_pandas_corr["features_raw"]], columns=feature_names)
y_pd = df_pandas_corr["median_house_value"]

correlations = X_pd.corrwith(y_pd).abs().sort_values(ascending=False)

print("Top correlated features with target:")
display(correlations.head(12).to_frame("abs_correlation"))


rf = RandomForestRegressor(
    featuresCol="features_raw",
    labelCol="median_house_value",
    numTrees=100,
    maxDepth=10,
    seed=42
)

rf_model = rf.fit(df_prepared)

# Extract & display feature importances
importances = rf_model.featureImportances
importance_list = [(feature_names[i], float(importances[i])) for i in range(len(importances))]
importance_df = spark.createDataFrame(importance_list, ["feature", "importance"]) \
                      .orderBy(col("importance").desc())

print("Random Forest Feature Importances (top 12):")
importance_df.show(12, truncate=False)

# Optional: GBT version (often gives different importance ranking)
gbt = GBTRegressor(featuresCol="features_raw", labelCol="median_house_value", maxIter=50, seed=42)
gbt_model = gbt.fit(df_prepared)
gbt_importance_df = spark.createDataFrame(
    [(feature_names[i], float(gbt_model.featureImportances[i])) for i in range(len(gbt_model.featureImportances))],
    ["feature", "importance"]
).orderBy(col("importance").desc())

print("Gradient Boosted Trees Feature Importances:")
gbt_importance_df.show(12, truncate=False)


# VarianceThresholdSelector (Spark 3.1+)
variance_selector = VarianceThresholdSelector(
    featuresCol="features_raw",
    outputCol="features_var_selected",
    threshold=0.01   # adjust based on your data scale
)

var_model = variance_selector.fit(df_prepared)
df_var_selected = var_model.transform(df_prepared)

print("Features kept after variance threshold:")
kept_indices = var_model.supportedIndices()
kept_features = [feature_names[i] for i in kept_indices]
print(kept_features)


# For illustration only – usually better for classification
chisq = ChiSqSelector(
    numTopFeatures=10,
    featuresCol="features_raw",
    outputCol="features_chisq",
    labelCol="median_house_value"   # works better if discretized
)

chisq_model = chisq.fit(df_prepared)
print("ChiSqSelector top features indices:", chisq_model.selectedFeatures)
print("Corresponding names:", [feature_names[i] for i in chisq_model.selectedFeatures])

top_features_from_rf = importance_df.limit(10).select("feature").rdd.flatMap(lambda x: x).collect()
print("Recommended features to keep:", top_features_from_rf)

