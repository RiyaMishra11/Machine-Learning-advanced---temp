"""223 — Tweedie Regression"""
from sklearn.datasets import make_regression
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def main():
    X, raw = make_regression(n_samples=1200, n_features=8, noise=20, random_state=42)
    y = np.maximum(np.exp((raw - raw.mean()) / (raw.std() * 2.5)), 0.05)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = make_pipeline(StandardScaler(), TweedieRegressor(power=1.5, alpha=0.5, max_iter=1000))
    model.fit(Xtr, ytr)
    pred = np.maximum(model.predict(Xte), 1e-9)
    print("Tweedie Regression (power=1.5)")
    print(f"MAE: {mean_absolute_error(yte,pred):.3f}")
    print(f"RMSE: {mean_squared_error(yte,pred)**0.5:.3f}")
    print(f"R²: {r2_score(yte,pred):.3f}")

if __name__ == "__main__":
    main()
