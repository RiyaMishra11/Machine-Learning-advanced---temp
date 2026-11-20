"""ML 62: Poisson Regression for Count Data."""
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_poisson_deviance, mean_absolute_error

def main():
    rng = np.random.default_rng(62)
    X = rng.normal(size=(900, 4))
    rate = np.exp(0.4 + 0.35*X[:, 0] - 0.25*X[:, 1] + 0.15*X[:, 2])
    y = rng.poisson(rate)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=62)
    model = PoissonRegressor(alpha=0.1, max_iter=1000)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("Poisson Regression")
    print(f"Mean Poisson deviance: {mean_poisson_deviance(yte, pred):.3f}")
    print(f"MAE: {mean_absolute_error(yte, pred):.3f}")
    print("Predicted counts:", np.round(pred[:5], 2))
    print("Actual counts:", yte[:5])

if __name__ == "__main__":
    main()
