"""
ML 38: Hierarchical (Agglomerative) Clustering
Install: pip install scikit-learn scipy matplotlib
"""
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main():
    X, _ = make_blobs(n_samples=250, centers=3, cluster_std=1.1, random_state=21)
    X_scaled = StandardScaler().fit_transform(X)

    # Ward linkage builds clusters by minimizing within-cluster variance.
    model = AgglomerativeClustering(n_clusters=3, linkage="ward")
    labels = model.fit_predict(X_scaled)
    print("Silhouette score:", round(silhouette_score(X_scaled, labels), 3))

    plt.figure(figsize=(8, 4))
    tree = linkage(X_scaled, method="ward")
    dendrogram(tree, truncate_mode="lastp", p=25)
    plt.title("Hierarchical Clustering Dendrogram (truncated)")
    plt.xlabel("Cluster or sample")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="viridis", s=25)
    plt.title("Agglomerative Clustering")
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
