"""ML 45: Isolation Forest anomaly detection. pip install scikit-learn matplotlib"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def main():
    X, _ = make_blobs(n_samples=450, centers=2, cluster_std=0.8, random_state=42)
    X = StandardScaler().fit_transform(X)
    pred = IsolationForest(contamination=0.06, random_state=42).fit_predict(X)
    out = pred == -1
    print("Anomalies:", int(out.sum()), "| Percentage:", round(out.mean() * 100, 2))
    plt.scatter(X[~out, 0], X[~out, 1], s=22, label="Normal")
    plt.scatter(X[out, 0], X[out, 1], s=35, marker="x", label="Anomaly")
    plt.title("Isolation Forest"); plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
