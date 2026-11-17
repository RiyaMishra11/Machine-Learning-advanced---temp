# ML 50: Polynomial Regression
# Install: pip install scikit-learn matplotlib numpy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def main():
    rng = np.random.default_rng(42)
    X = np.linspace(-3, 3, 180).reshape(-1, 1)
    y = 1.5 + 2*X[:, 0] - 0.8*X[:, 0]**2 + 0.25*X[:, 0]**3
    y += rng.normal(0, 1, len(y))
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("MSE:", round(mean_squared_error(yte, pred), 3))
    print("R2:", round(r2_score(yte, pred), 3))
    order = np.argsort(Xte[:, 0])
    plt.scatter(Xte[:, 0], yte, label="Actual")
    plt.plot(Xte[order, 0], pred[order], label="Prediction")
    plt.title("Polynomial Regression (degree 3)")
    plt.xlabel("X"); plt.ylabel("y"); plt.legend(); plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
