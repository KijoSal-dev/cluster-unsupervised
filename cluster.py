# Import libraries for numerical operations, plotting, data generation, and clustering.
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.datasets import make_blobs  
from sklearn.cluster import KMeans  

# Generate synthetic data with three distinct clusters.
X, _ = make_blobs(n_samples=300,     # Generate 300 points.
                  centers=3,         # Create 3 clusters.
                  cluster_std=0.60,    # Standard deviation of clusters.
                  random_state=42)     # For reproducibility.

# Initialize K-Means clustering algorithm with 3 clusters.
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)  # Apply the algorithm to the data.

# Extract the predicted cluster labels for each data point.
clusters = kmeans.labels_

# Visualize the clustering result.
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', edgecolor='k')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()