"""ML 65: Stacking Regression."""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def main():
    X, y = make_regression(n_samples=800, n_features=9, noise=24, random_state=65)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=65)
    estimators = [
        ("forest", RandomForestRegressor(n_estimators=100, random_state=65, n_jobs=-1)),
        ("extra_trees", ExtraTreesRegressor(n_estimators=100, random_state=65, n_jobs=-1)),
        ("svr", make_pipeline(StandardScaler(), SVR(C=10))),
    ]
    model = StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=1), cv=5, n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("Stacking Regression")
    print(f"MAE: {mean_absolute_error(yte, pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(yte, pred)):.3f}")
    print(f"R2: {r2_score(yte, pred):.3f}")

if __name__ == "__main__":
    main()
