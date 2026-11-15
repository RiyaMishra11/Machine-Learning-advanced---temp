"""ML 47: One-Class SVM anomaly detection. pip install scikit-learn matplotlib"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

def main():
    X, _ = make_blobs(n_samples=400, centers=1, cluster_std=0.7, random_state=42)
    X = StandardScaler().fit_transform(X)
    model = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05)
    pred = model.fit_predict(X)
    out = pred == -1
    print("Potential anomalies:", int(out.sum()))
    print("Decision scores:", model.decision_function(X[:10]).round(3))
    plt.scatter(X[~out, 0], X[~out, 1], s=22, label="Inlier")
    plt.scatter(X[out, 0], X[out, 1], s=38, marker="x", label="Potential anomaly")
    plt.title("One-Class SVM"); plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
