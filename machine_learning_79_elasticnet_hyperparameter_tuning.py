"""ML 79: Tune ElasticNet regression with GridSearchCV."""
from sklearn.datasets import make_regression
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():
    X, y = make_regression(n_samples=500, n_features=15, noise=18,
                           random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    pipe = make_pipeline(StandardScaler(), ElasticNet(max_iter=20000))
    params = {
        "elasticnet__alpha": [0.01, 0.1, 1.0, 10.0],
        "elasticnet__l1_ratio": [0.2, 0.5, 0.8, 1.0],
    }
    search = GridSearchCV(pipe, params, cv=5, scoring="neg_mean_absolute_error",
                          n_jobs=-1)
    search.fit(X_train, y_train)
    pred = search.predict(X_test)
    print("Best parameters:", search.best_params_)
    print("Test MAE:", round(mean_absolute_error(y_test, pred), 2))
    print("Test R²:", round(r2_score(y_test, pred), 3))

if __name__ == "__main__":
    main()
