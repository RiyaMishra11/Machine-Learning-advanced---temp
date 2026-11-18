"""ML 54: Random Forest Regression | pip install scikit-learn matplotlib"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    X, y = make_regression(n_samples=500, n_features=6, n_informative=5, noise=22, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    model = RandomForestRegressor(n_estimators=180, max_depth=8, random_state=42, n_jobs=-1).fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("MAE:", round(mean_absolute_error(yte, pred), 3), "R2:", round(r2_score(yte, pred), 3))
    print("Feature importances:", np.round(model.feature_importances_, 3))
    plt.scatter(yte, pred, alpha=.7)
    lo, hi = min(yte.min(), pred.min()), max(yte.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], "--", label="Perfect prediction")
    plt.title("Random Forest: Actual vs Predicted"); plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
