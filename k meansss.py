import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Function to plot clustered data
def plot_clusters(X_pca, labels, centroids_pca=None, title="Clustering Results"):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', edgecolors='k', alpha=0.6)
    if centroids_pca is not None:
        plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, c='red', label='Centroids')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(title)
    plt.legend()
    plt.show()

# Load the Iris dataset
iris = load_iris()
X_iris = iris.data

# Standardize features
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)

# Finding the Best k using the Elbow Method
inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_iris_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Method
plt.figure(figsize=(8, 6))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k (Iris Dataset)')
plt.show()

# Apply K-Means with k=3 (as found previously)
kmeans_iris = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_iris = kmeans_iris.fit_predict(X_iris_scaled)

# Reduce dimensions using PCA for visualization
pca = PCA(n_components=2)
X_iris_pca = pca.fit_transform(X_iris_scaled)
centroids_iris_pca = pca.transform(kmeans_iris.cluster_centers_)

# Plot K-Means clustering results on Iris dataset
plot_clusters(X_iris_pca, labels_iris, centroids_iris_pca, title="K-Means Clustering on Iris Dataset")

# ---------------------------------------
# Using K-Means on the Digits Dataset
# ---------------------------------------

# Load Digits dataset
digits = load_digits()
X_digits = digits.data

# Standardize features
X_digits_scaled = scaler.fit_transform(X_digits)

# Finding best k using Elbow Method on Digits dataset
inertia_digits = []
k_values_digits = range(1, 11)

for k in k_values_digits:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_digits_scaled)
    inertia_digits.append(kmeans.inertia_)

# Plot Elbow Method for Digits dataset
plt.figure(figsize=(8, 6))
plt.plot(k_values_digits, inertia_digits, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k (Digits Dataset)')
plt.show()

# Apply K-Means with k=10 (since Digits has 10 digits)
kmeans_digits = KMeans(n_clusters=10, random_state=42, n_init=10)
labels_digits = kmeans_digits.fit_predict(X_digits_scaled)

# Reduce dimensions using PCA for visualization
pca_digits = PCA(n_components=2)
X_digits_pca = pca_digits.fit_transform(X_digits_scaled)
centroids_digits_pca = pca_digits.transform(kmeans_digits.cluster_centers_)

# Plot K-Means clustering results on Digits dataset
plot_clusters(X_digits_pca, labels_digits, centroids_digits_pca, title="K-Means Clustering on Digits Dataset")

# ---------------------------------------
# Compare K-Means with DBSCAN
# ---------------------------------------

# Apply DBSCAN on the Iris dataset
dbscan_iris = DBSCAN(eps=1.0, min_samples=5)  # Tune parameters
dbscan_labels_iris = dbscan_iris.fit_predict(X_iris_scaled)

# Plot DBSCAN clustering results on Iris dataset
plot_clusters(X_iris_pca, dbscan_labels_iris, title="DBSCAN Clustering on Iris Dataset")

# Count unique clusters (excluding noise)
unique_dbscan_iris = set(dbscan_labels_iris)
num_clusters_dbscan_iris = len(unique_dbscan_iris) - (1 if -1 in unique_dbscan_iris else 0)

print(f"DBSCAN found {num_clusters_dbscan_iris} clusters (excluding noise) in the Iris dataset.")

# Apply DBSCAN on Digits dataset
dbscan_digits = DBSCAN(eps=2.0, min_samples=10)  # Tune parameters
dbscan_labels_digits = dbscan_digits.fit_predict(X_digits_scaled)

# Plot DBSCAN clustering results on Digits dataset
plot_clusters(X_digits_pca, dbscan_labels_digits, title="DBSCAN Clustering on Digits Dataset")

# Count unique clusters (excluding noise)
unique_dbscan_digits = set(dbscan_labels_digits)
num_clusters_dbscan_digits = len(unique_dbscan_digits) - (1 if -1 in unique_dbscan_digits else 0)

print(f"DBSCAN found {num_clusters_dbscan_digits} clusters (excluding noise) in the Digits dataset.")
