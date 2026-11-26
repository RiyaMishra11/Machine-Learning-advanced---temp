"""ML 84: Advanced Stacking Regression with diverse base estimators."""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

def main():
    X, y = make_regression(n_samples=900, n_features=14, noise=28, random_state=84)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=84)
    base_models = [
        ("random_forest", RandomForestRegressor(n_estimators=150, random_state=84,
                                                n_jobs=-1)),
        ("extra_trees", ExtraTreesRegressor(n_estimators=150, random_state=84,
                                            n_jobs=-1)),
        ("svr", make_pipeline(StandardScaler(), SVR(C=30, epsilon=0.2))),
    ]
    model = StackingRegressor(estimators=base_models,
                              final_estimator=Ridge(alpha=1.0), cv=5, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Advanced Stacking Regression")
    print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
    print(f"R2: {r2_score(y_test, pred):.3f}")
    print("First predictions:", np.round(pred[:5], 2))

if __name__ == "__main__":
    main()
