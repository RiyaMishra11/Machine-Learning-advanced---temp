"""ML 68: Learning Curve for Regression."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import learning_curve, KFold

def main():
    X, y = make_regression(n_samples=800, n_features=10, noise=25, random_state=68)
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=68, n_jobs=-1)
    cv = KFold(n_splits=5, shuffle=True, random_state=68)
    sizes, train_scores, valid_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 8),
        cv=cv, scoring="r2", n_jobs=-1
    )
    train_mean, valid_mean = train_scores.mean(axis=1), valid_scores.mean(axis=1)
    print("Learning Curve Regression")
    for size, tr, va in zip(sizes, train_mean, valid_mean):
        print(f"Samples: {size:4d} | Train R2: {tr:.3f} | CV R2: {va:.3f}")
    plt.plot(sizes, train_mean, marker="o", label="Training R2")
    plt.plot(sizes, valid_mean, marker="o", label="Cross-validation R2")
    plt.xlabel("Number of training samples")
    plt.ylabel("R2 score")
    plt.title("Regression Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
