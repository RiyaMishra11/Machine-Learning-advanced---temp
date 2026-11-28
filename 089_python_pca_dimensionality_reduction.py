"""
File 089: Principal Component Analysis (PCA)
Topic: Reduce the number of features while retaining as much information as possible.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load a sample dataset
X, y = load_iris(return_X_y=True)

# PCA is scale-sensitive, so standardize the features first
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce the four original features to two principal components
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

print("Original shape:", X.shape)
print("Reduced shape:", X_reduced.shape)
print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance retained:", round(pca.explained_variance_ratio_.sum(), 4))
print("First five transformed rows:\n", X_reduced[:5])
