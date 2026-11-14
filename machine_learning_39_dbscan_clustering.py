"""
ML 39: DBSCAN Clustering
DBSCAN can discover irregularly shaped clusters and label outliers as -1.
Install: pip install scikit-learn matplotlib
"""
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def main():
    X, _ = make_moons(n_samples=500, noise=0.07, random_state=42)
    X_scaled = StandardScaler().fit_transform(X)

    model = DBSCAN(eps=0.25, min_samples=5)
    labels = model.fit_predict(X_scaled)

    cluster_labels = set(labels) - {-1}
    noise_count = int((labels == -1).sum())
    print("Clusters found:", len(cluster_labels))
    print("Noise points:", noise_count)

    # Silhouette score is meaningful only when at least two clusters remain.
    if len(cluster_labels) >= 2:
        keep = labels != -1
        print("Silhouette score (excluding noise):",
              round(silhouette_score(X_scaled[keep], labels[keep]), 3))
    else:
        print("Silhouette score skipped: fewer than two non-noise clusters.")

    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap="plasma", s=22)
    plt.title("DBSCAN Clustering (-1 indicates noise)")
    plt.xlabel("Feature 1 (scaled)")
    plt.ylabel("Feature 2 (scaled)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
