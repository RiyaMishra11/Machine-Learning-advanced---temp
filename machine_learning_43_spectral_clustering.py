"""ML 43: Spectral Clustering. Install: scikit-learn matplotlib"""
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def main():
    X, _ = make_moons(n_samples=450, noise=0.06, random_state=42)
    X = StandardScaler().fit_transform(X)
    model = SpectralClustering(n_clusters=2, affinity="nearest_neighbors",
                               n_neighbors=12, assign_labels="kmeans", random_state=42)
    labels = model.fit_predict(X)
    print("Silhouette score:", round(silhouette_score(X, labels), 3))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=24)
    plt.title("Spectral Clustering on Two-Moons Data")
    plt.xlabel("Feature 1 (scaled)"); plt.ylabel("Feature 2 (scaled)")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
