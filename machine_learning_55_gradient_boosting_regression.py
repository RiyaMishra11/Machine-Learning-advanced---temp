"""ML 55: Gradient Boosting Regression | pip install scikit-learn matplotlib"""
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    X, y = make_regression(n_samples=500, n_features=8, noise=20, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=150, learning_rate=.05, max_depth=2,
                                      random_state=42).fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("MSE:", round(mean_squared_error(yte, pred), 3), "R2:", round(r2_score(yte, pred), 3))
    plt.scatter(yte, pred, alpha=.7)
    lo, hi = min(yte.min(), pred.min()), max(yte.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], "--", label="Perfect prediction")
    plt.title("Gradient Boosting Regression"); plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
