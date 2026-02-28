from pyspark.ml.feature import Imputer

# Define the columns you want to fix
target_cols = ["col_a", "col_b"]

# Map inputCols and outputCols to the same list to overwrite
imputer = Imputer(
    inputCols=target_cols, 
    outputCols=target_cols, 
    strategy="mean"
)

# Fit and transform
df = imputer.fit(df).transform(df)
