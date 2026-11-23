"""ML 76: Regression Feature Selection with SelectFromModel."""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    X, y = make_regression(n_samples=900, n_features=20, n_informative=6,
                           noise=20, random_state=76)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=76)
    selector = SelectFromModel(
        RandomForestRegressor(n_estimators=120, random_state=76, n_jobs=-1),
        threshold="median")
    model = make_pipeline(selector, Ridge(alpha=1.0))
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    support = model.named_steps["selectfrommodel"].get_support()
    print("Regression Feature Selection")
    print("Features before:", X.shape[1])
    print("Features selected:", int(support.sum()))
    print("Selected indices:", np.flatnonzero(support).tolist())
    print(f"MAE: {mean_absolute_error(yte, pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(yte, pred)):.3f}")
    print(f"R2: {r2_score(yte, pred):.3f}")

if __name__ == "__main__":
    main()
