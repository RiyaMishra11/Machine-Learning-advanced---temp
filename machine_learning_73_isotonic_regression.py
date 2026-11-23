"""ML 73: Isotonic Regression."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    rng = np.random.default_rng(73)
    X = np.sort(rng.uniform(0, 10, 500))
    y = 2.5 * X + 0.7 * np.sin(X) + rng.normal(0, 1.8, X.size)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=73)
    model = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("Isotonic Regression")
    print(f"MAE: {mean_absolute_error(yte, pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(yte, pred)):.3f}")
    print(f"R2: {r2_score(yte, pred):.3f}")
    order = np.argsort(Xte)
    print("Predictions monotonic:", bool(np.all(np.diff(model.predict(Xte[order])) >= -1e-10)))

if __name__ == "__main__":
    main()
