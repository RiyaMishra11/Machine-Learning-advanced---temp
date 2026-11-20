"""ML 63: Tweedie Regression."""
import numpy as np
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_tweedie_deviance, mean_absolute_error

def main():
    rng = np.random.default_rng(63)
    X = rng.normal(size=(1000, 5))
    mean = np.exp(1.0 + 0.25*X[:, 0] - 0.18*X[:, 1] + 0.12*X[:, 2])
    y = rng.gamma(shape=2.0, scale=mean/2.0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=63)
    model = TweedieRegressor(power=1.5, alpha=0.1, link="log", max_iter=1000)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("Tweedie Regression (power=1.5)")
    print(f"Mean Tweedie deviance: {mean_tweedie_deviance(yte, pred, power=1.5):.3f}")
    print(f"MAE: {mean_absolute_error(yte, pred):.3f}")
    print("Predictions:", np.round(pred[:5], 2))

if __name__ == "__main__":
    main()
