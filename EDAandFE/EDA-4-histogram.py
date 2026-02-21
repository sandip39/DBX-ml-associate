import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# 1. Load the housing data
# Fetch the California housing dataset from scikit-learn
# The data is returned as a dictionary-like object (Bunch)
housing_bunch = fetch_california_housing(as_frame=True)

# Extract the data into a pandas DataFrame and target into a Series
# The 'MedHouseVal' is the median house value in hundreds of thousands of dollars
df_housing = housing_bunch.frame
median_house_value = df_housing['MedHouseVal']

# 2. Draw a histogram plot of the median house value
plt.figure(figsize=(8, 6))
plt.hist(median_house_value, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Median House Value in California')
plt.xlabel('Median House Value (hundreds of thousands of $)')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.7)

# Display the plot
plt.show()
