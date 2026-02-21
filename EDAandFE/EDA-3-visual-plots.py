import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# 1. Load the housing data
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Calculate the correlation matrix
corr_matrix = df.corr()

# 3. Create the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap for California Housing Data')
plt.show()


# Select a few key features for better visibility
features = ['MedInc', 'HouseAge', 'AveRooms', 'MedHouseVal']
sns.pairplot(df[features], diag_kind='hist', corner=True)
plt.suptitle('Pair Plot of Housing Features', y=1.02)
plt.show()
