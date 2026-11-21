"""ML 67: Randomized Search Hyperparameter Tuning for Regression."""
import numpy as np
from scipy.stats import randint
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    X, y = make_regression(n_samples=700, n_features=8, noise=22, random_state=67)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=67)
    model = RandomForestRegressor(random_state=67, n_jobs=-1)
    params = {
        "n_estimators": randint(80, 220),
        "max_depth": [None, 5, 8, 12, 16],
        "min_samples_split": randint(2, 10),
        "min_samples_leaf": randint(1, 5),
        "max_features": [1.0, "sqrt", 0.8],
    }
    search = RandomizedSearchCV(model, params, n_iter=20, cv=5, scoring="r2",
                                random_state=67, n_jobs=-1)
    search.fit(Xtr, ytr)
    pred = search.best_estimator_.predict(Xte)
    print("Randomized Search Regression")
    print("Best parameters:", search.best_params_)
    print(f"Best CV R2: {search.best_score_:.3f}")
    print(f"Test MAE: {mean_absolute_error(yte, pred):.3f}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(yte, pred)):.3f}")
    print(f"Test R2: {r2_score(yte, pred):.3f}")

if __name__ == "__main__":
    main()
