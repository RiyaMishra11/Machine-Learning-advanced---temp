"""
ML 40: Clustering Evaluation
Compare clustering solutions using internal evaluation metrics.
Install: pip install scikit-learn
"""
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler


def evaluate_clustering(X, labels, name):
    unique_labels = set(labels)
    if len(unique_labels) < 2 or len(unique_labels) >= len(labels):
        print(f"{name}: metrics skipped (need 2 to n-1 clusters).")
        return

    print(f"\n{name}")
    print("Silhouette score (higher is better):",
          round(silhouette_score(X, labels), 3))
    print("Davies-Bouldin index (lower is better):",
          round(davies_bouldin_score(X, labels), 3))
    print("Calinski-Harabasz score (higher is better):",
          round(calinski_harabasz_score(X, labels), 3))


def main():
    X, _ = make_blobs(n_samples=450, centers=4, cluster_std=1.2, random_state=42)
    X = StandardScaler().fit_transform(X)

    for k in (3, 4, 5):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
        evaluate_clustering(X, labels, f"K-Means with k={k}")


if __name__ == "__main__":
    main()
