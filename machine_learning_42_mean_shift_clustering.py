"""ML 42: Mean Shift clustering. Install: scikit-learn matplotlib"""
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

def main():
    X, _ = make_blobs(n_samples=350, centers=4, cluster_std=1.0, random_state=17)
    X = StandardScaler().fit_transform(X)
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=min(300, len(X)))
    if bandwidth <= 0:
        raise ValueError("Estimated bandwidth is not positive; try another quantile.")
    model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    labels = model.fit_predict(X)
    print("Bandwidth:", round(float(bandwidth), 3))
    print("Clusters found:", len(model.cluster_centers_))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="plasma", s=24)
    plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
                marker="X", s=180, c="black", label="Cluster centers")
    plt.title("Mean Shift Clustering"); plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
