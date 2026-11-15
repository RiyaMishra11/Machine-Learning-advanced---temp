"""ML 44: Tune DBSCAN parameters with silhouette score. Install: scikit-learn pandas"""
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def main():
    X, _ = make_moons(n_samples=450, noise=0.07, random_state=42)
    X = StandardScaler().fit_transform(X)
    results = []
    for eps in (0.18, 0.22, 0.26, 0.30, 0.34):
        for min_samples in (4, 5, 7):
            labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X)
            keep = labels != -1
            clusters = set(labels[keep])
            score = None
            if len(clusters) >= 2 and keep.sum() > len(clusters):
                score = silhouette_score(X[keep], labels[keep])
            results.append({"eps": eps, "min_samples": min_samples,
                            "clusters": len(clusters),
                            "noise_points": int((labels == -1).sum()),
                            "silhouette_score": score})
    table = pd.DataFrame(results).sort_values(
        "silhouette_score", ascending=False, na_position="last")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    valid = table.dropna(subset=["silhouette_score"])
    if not valid.empty:
        print("\nBest tested setting:\n", valid.iloc[0].to_string())
    else:
        print("\nNo tested setting produced a valid silhouette score.")

if __name__ == "__main__":
    main()
