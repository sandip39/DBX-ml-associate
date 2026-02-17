
# Initialize Spark Session
spark = SparkSession.builder.appName("HousingEDA").getOrCreate()

# Define the path to your housing data CSV file
# Replace 'housing_data.csv' with the actual path to your file
csv_file_path = "housing_data.csv"

# Load the CSV data into a Spark DataFrame
# 'header=True' treats the first row as column names
# 'inferSchema=True' automatically infers data types
spark_df = spark.read.csv(csv_file_path, header=True, inferSchema=True)

# Display the first few rows and the schema to verify loading
print("Spark DataFrame Head:")
spark_df.show(5)
print("Spark DataFrame Schema:")
spark_df.printSchema()

spark_df.describe().show()

spark_df.groupBy("ocean_proximity").count().show()

spark_df.groupBy("ocean_proximity").mean().show()

spark_df.groupBy("ocean_proximity").max().show()

spark_df.groupBy("ocean_proximity").min().show()

spark_df.groupBy("ocean_proximity").sum().show()

# Convert the Spark DataFrame to a pandas DataFrame
# Consider using sampling (e.g., spark_df.sample(0.1)) for very large datasets
pandas_df = spark_df.toPandas()

print("Pandas DataFrame Head:")
print(pandas_df.head())


# Display descriptive statistics
print("\nDescriptive Statistics:")
print(pandas_df.describe())

# Check for missing values
print("\nMissing Values:")
print(pandas_df.isnull().sum())

pandas_df.info()
pandas_df.count()
pandas_df.columns
pandas_df.head()
pandas_df.tail()
pandas_df.shape
pandas_df.dtypes
pandas_df.index
pandas_df.values
pandas_df.describe()


pandas_df.corr()



