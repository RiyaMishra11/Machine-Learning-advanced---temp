"""222 — Gamma Regression"""
from sklearn.datasets import make_regression
from sklearn.linear_model import GammaRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_gamma_deviance, mean_absolute_error
import numpy as np

def main():
    X, raw = make_regression(n_samples=1200, n_features=8, noise=25, random_state=42)
    y = np.exp((raw - raw.mean()) / (raw.std() * 3)) + 0.5
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(StandardScaler(), GammaRegressor(alpha=1.0, max_iter=1000))
    model.fit(Xtr, ytr)
    pred = np.maximum(model.predict(Xte), 1e-9)
    print("Gamma Regression")
    print(f"MAE: {mean_absolute_error(yte,pred):.3f}")
    print(f"Mean Gamma Deviance: {mean_gamma_deviance(yte,pred):.3f}")

if __name__ == "__main__":
    main()
