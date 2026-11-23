"""ML 75: Extra Trees Regression and feature importance."""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    X, y = make_regression(n_samples=850, n_features=10, n_informative=6,
                           noise=24, random_state=75)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=75)
    model = ExtraTreesRegressor(n_estimators=180, min_samples_leaf=2,
                                max_features=0.9, random_state=75, n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("Extra Trees Regression")
    print(f"MAE: {mean_absolute_error(yte, pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(yte, pred)):.3f}")
    print(f"R2: {r2_score(yte, pred):.3f}")
    print("Feature importances:")
    for idx in np.argsort(model.feature_importances_)[::-1]:
        print(f"Feature {idx}: {model.feature_importances_[idx]:.3f}")

if __name__ == "__main__":
    main()
