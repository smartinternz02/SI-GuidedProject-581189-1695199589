import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('C:/Users/Sumit/AI ML SmartBridge/Assignment-5/dataset/Mall_Customers.csv')

# Drop the 'Gender' column
df.drop('Gender', axis=1, inplace=True)

# Normalize the remaining variables
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)

from sklearn.cluster import KMeans

# Create a KMeans object with 3 clusters
kmeans = KMeans(n_clusters=3)

# Fit the KMeans model to the data
kmeans.fit(scaled_df)

# Predict the cluster labels for each customer
cluster_labels = kmeans.predict(scaled_df)

# Add the cluster labels to the original DataFrame
df['Cluster Label'] = cluster_labels

import matplotlib.pyplot as plt

# Create a scatter plot of the customers' spending scores and annual incomes, colored by cluster label
plt.scatter(df['Spending Score'], df['Annual Income'], c=df['Cluster Label'])
plt.xlabel('Spending Score')
plt.ylabel('Annual Income')
plt.title('Mall Customer Segmentation')
plt.show()

# Calculate some descriptive statistics for each cluster
cluster_stats = df.groupby('Cluster Label').agg(['mean', 'std'])
print(cluster_stats)
