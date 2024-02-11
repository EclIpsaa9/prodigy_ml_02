#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# In[6]:


data = pd.read_csv('C:/Users/DANICA/Desktop/prodigy/Mall_Customers.csv') 


# In[7]:


X = data.iloc[:, [3, 4]].values 


# In[8]:


scaler = StandardScaler()
X = scaler.fit_transform(X)


# In[9]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# In[10]:


plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  # Within-Cluster Sum of Squares
plt.show()


# In[11]:


optimal_num_clusters = 5 


# In[12]:


kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
kmeans.fit(X)


# In[13]:


data['Cluster'] = kmeans.labels_


# In[14]:


plt.scatter(X[data['Cluster'] == 0, 0], X[data['Cluster'] == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[data['Cluster'] == 1, 0], X[data['Cluster'] == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(X[data['Cluster'] == 2, 0], X[data['Cluster'] == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(X[data['Cluster'] == 3, 0], X[data['Cluster'] == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[data['Cluster'] == 4, 0], X[data['Cluster'] == 4, 1], s=100, c='magenta', label='Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (Scaled)')
plt.ylabel('Spending Score (Scaled)')
plt.legend()
plt.show()


# In[ ]:




