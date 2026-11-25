"""
ML 79: ElasticNet Hyperparameter Tuning
Compares candidate alpha and l1_ratio values using GridSearchCV.
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    X, y = make_regression(
        n_samples=500, n_features=18, n_informative=10,
        noise=22, random_state=79
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=79
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", ElasticNet(max_iter=20000, random_state=79)),
    ])
    parameter_grid = {
        "model__alpha": [0.01, 0.1, 1.0, 10.0],
        "model__l1_ratio": [0.2, 0.5, 0.8, 1.0],
    }
    search = GridSearchCV(
        pipeline, parameter_grid, cv=5,
        scoring="neg_mean_absolute_error", n_jobs=-1
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    print("ElasticNet Hyperparameter Tuning")
    print("Best parameters:", search.best_params_)
    print(f"Best CV MAE: {-search.best_score_:.2f}")
    print(f"Test MAE:    {mean_absolute_error(y_test, predictions):.2f}")
    print(f"Test R²:     {r2_score(y_test, predictions):.3f}")


if __name__ == "__main__":
    main()
