"""ML 81: Histogram-Based Gradient Boosting Regression."""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def main():
    X, y = make_regression(n_samples=1000, n_features=14, noise=25, random_state=81)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=81)
    model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.08,
                                           max_leaf_nodes=20, early_stopping=True,
                                           random_state=81)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Histogram Gradient Boosting Regression")
    print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.2f}")
    print(f"R2: {r2_score(y_test, pred):.3f}")
    print("Iterations:", model.n_iter_)

if __name__ == "__main__":
    main()
