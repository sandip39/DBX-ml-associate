import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing

# 1. Load the housing dataset
# The California housing dataset is available in the scikit-learn library
housing = fetch_california_housing()

# Convert the data into a pandas DataFrame for easier manipulation and plotting
housing_df = pd.DataFrame(data=housing.data, columns=housing.feature_names)
housing_df['HousePrice'] = housing.target # Add the target variable (median house value) as a column

# 2. Plot a scatter plot between Median Income ('MedInc') and Median House Value ('HousePrice')
plt.figure(figsize=(8, 6))
plt.scatter(housing_df['MedInc'], housing_df['HousePrice'], alpha=0.4)
plt.title('Scatter Plot: Median House Value vs. Median Income')
plt.xlabel('Median Income (in tens of thousands of USD)')
plt.ylabel('Median House Value (in hundreds of thousands of USD)')
plt.grid(True)

# 3. Display the plot in the Databricks notebook
# Databricks automatically displays matplotlib plots, but plt.show() ensures rendering in all environments
plt.show()
