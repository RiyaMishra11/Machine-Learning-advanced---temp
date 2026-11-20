"""ML 64: Regression Residual Diagnostics."""
import numpy as np
from sklearn.datasets import make_friedman1
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    X, y = make_friedman1(n_samples=900, n_features=8, noise=2.0, random_state=64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=64)
    model = RandomForestRegressor(n_estimators=160, min_samples_leaf=2,
                                  random_state=64, n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    residuals = yte - pred
    print("Regression Residual Diagnostics")
    print(f"MAE: {mean_absolute_error(yte, pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(yte, pred)):.3f}")
    print(f"R2: {r2_score(yte, pred):.3f}")
    print(f"Mean residual: {np.mean(residuals):.3f}")
    print(f"Residual standard deviation: {np.std(residuals):.3f}")
    print(f"Residual 5th/95th percentiles: {np.percentile(residuals, 5):.3f} / {np.percentile(residuals, 95):.3f}")
    print("Top feature importances:")
    for i in np.argsort(model.feature_importances_)[::-1][:5]:
        print(f"Feature {i}: {model.feature_importances_[i]:.3f}")

if __name__ == "__main__":
    main()
