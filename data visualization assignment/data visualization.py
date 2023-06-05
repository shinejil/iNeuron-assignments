#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Perform dimensionality reduction to 3 dimensions using PCA
pca = PCA(n_components=3)
X_transformed = pca.fit_transform(X)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the transformed data points with color-coded classes
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=y)

# Set labels and title
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('Iris Data in 3D')

# Show the plot
plt.show()

