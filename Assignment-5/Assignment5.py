# import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('C:/Users/Sumit/Downloads/Assignment-5/Assignment-5/Mall_Customers.csv')
df.head()
df.shape
df.info()
df.isnull().sum()
df.describe()
from sklearn import cluster
new_df = df.iloc[:,-2:]
new_df.head()
error=[]
for i in range(1,11):
  kmeans = cluster.KMeans(n_clusters=i,init = 'k-means++',random_state=4)
  kmeans.fit(new_df)
  error.append(kmeans.inertia_)

plt.plot(range(1,11),error)
plt.title('The Elbow point graph')
plt.xlabel('number of clusters')
plt.ylabel('error')
plt.show()

km_model = cluster.KMeans(n_clusters=5,init = 'k-means++',random_state=0)
km_model.fit(new_df)

pred = km_model.predict(new_df)
pred

# Testing the model with random observation

km_model.predict([[60,50]])
km_model.predict([[15,1]])
km_model.predict([[41,34]])
km_model.predict([[137,99]])
km_model.predict([[78,73]])