"""ML 53: Decision Tree Regression | pip install scikit-learn matplotlib"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def main():
    X, y = make_regression(n_samples=350, n_features=1, noise=18, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=42)
    model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=42).fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("MSE:", round(mean_squared_error(yte, pred), 3), "R2:", round(r2_score(yte, pred), 3))
    grid = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    plt.scatter(Xte[:, 0], yte, label="Actual")
    plt.plot(grid[:, 0], model.predict(grid), label="Tree prediction")
    plt.title("Decision Tree Regression"); plt.xlabel("Feature"); plt.ylabel("Target")
    plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
