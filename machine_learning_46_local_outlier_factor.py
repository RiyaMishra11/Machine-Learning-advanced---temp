"""ML 46: Local Outlier Factor. pip install scikit-learn matplotlib"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

def main():
    X, _ = make_moons(n_samples=450, noise=0.08, random_state=42)
    X = StandardScaler().fit_transform(X)
    model = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
    pred = model.fit_predict(X)
    out = pred == -1
    print("Outliers detected:", int(out.sum()))
    print("Negative outlier factors (first 10):", model.negative_outlier_factor_[:10].round(3))
    plt.scatter(X[~out, 0], X[~out, 1], s=22, label="Inlier")
    plt.scatter(X[out, 0], X[out, 1], s=38, marker="x", label="Outlier")
    plt.title("Local Outlier Factor"); plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
