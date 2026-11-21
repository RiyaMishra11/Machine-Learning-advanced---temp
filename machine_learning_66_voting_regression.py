"""ML 66: Voting Regression."""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    X, y = make_regression(n_samples=750, n_features=8, noise=20, random_state=66)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=66)
    estimators = [
        ("ridge", Ridge(alpha=1)),
        ("forest", RandomForestRegressor(n_estimators=100, random_state=66, n_jobs=-1)),
        ("gradient_boosting", GradientBoostingRegressor(n_estimators=100, random_state=66)),
        ("extra_trees", ExtraTreesRegressor(n_estimators=100, random_state=66, n_jobs=-1)),
    ]
    model = VotingRegressor(estimators=estimators, weights=[1, 2, 2, 2], n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("Voting Regression")
    print(f"MAE: {mean_absolute_error(yte, pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(yte, pred)):.3f}")
    print(f"R2: {r2_score(yte, pred):.3f}")

if __name__ == "__main__":
    main()
