# Define paths and table name
source_path = "/databricks-datasets/iot-drift/data-turbines/turbine-data.csv"
table_name = "main.default.wind_turbine_data"

# Read raw CSV and save to Unity Catalog
df_wind = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(source_path)
df_wind.write.mode("overwrite").format("delta").saveAsTable(table_name)


import mlflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

# Enable autologging for TensorFlow/Keras
mlflow.tensorflow.autolog()

#  for training
data = spark.table(table_name).toPandas()
X = data.drop(["power"], axis=1) # Replace with actual target column
y = data["power"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build a simple Keras model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

with mlflow.start_run(run_name="Wind_Farm_Keras") as run:
    model.fit(X_train, y_train, epochs=10, validation_split=0.2)
    run_id = run.info.run_id
model_name = "main.default.wind_power_forecasting"
model_uri = f"runs:/{run_id}/model"

# Register the model version in Unity Catalog
result = mlflow.register_model(model_uri, model_name)

from pyspark.sql.functions import struct

# Load the registered model from Unity Catalog
# Use "latest" version or a specific alias like "@prod"
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{model_name}/1")

# Load new data for scoring
df_new_data = spark.table(table_name) # Or a different table with new sensor readings

# Perform distributed batch inference
df_predictions = df_new_data.withColumn(
    "predicted_power", 
    predict_udf(struct(*df_new_data.columns))
)

# Save the results back to a prediction table
df_predictions.write.mode("append").saveAsTable("main.default.wind_power_predictions")
