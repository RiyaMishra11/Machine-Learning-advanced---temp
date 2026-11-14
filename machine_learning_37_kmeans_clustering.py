"""
ML 37: K-Means Clustering
Unsupervised learning example using synthetic data.
Install: pip install scikit-learn matplotlib
"""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main():
    X, _ = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)

    model = KMeans(n_clusters=4, n_init=10, random_state=42)
    labels = model.fit_predict(X_scaled)

    print("Inertia:", round(model.inertia_, 3))
    print("Silhouette score:", round(silhouette_score(X_scaled, labels), 3))

    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="viridis", s=25)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                c="red", marker="X", s=180, label="Centroids")
    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
