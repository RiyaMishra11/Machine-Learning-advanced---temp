"""ML 41: Gaussian Mixture Model clustering. Install: scikit-learn matplotlib"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def main():
    X, _ = make_blobs(n_samples=500, centers=3, cluster_std=1.3, random_state=42)
    X = StandardScaler().fit_transform(X)
    model = GaussianMixture(n_components=3, covariance_type="full", random_state=42)
    labels = model.fit_predict(X)
    print("AIC:", round(model.aic(X), 2), "BIC:", round(model.bic(X), 2))
    print("First five membership probabilities:\n", model.predict_proba(X)[:5].round(3))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=24)
    plt.title("Gaussian Mixture Model Clustering")
    plt.xlabel("Feature 1 (scaled)"); plt.ylabel("Feature 2 (scaled)")
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
