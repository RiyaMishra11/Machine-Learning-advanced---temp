# ML 52: Regression Evaluation Metrics
# Install: pip install scikit-learn
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error, median_absolute_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def evaluate(name, model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    print("\\n" + name)
    print("MAE:", round(mean_absolute_error(yte, pred), 3))
    print("MSE:", round(mean_squared_error(yte, pred), 3))
    print("RMSE:", round(mean_squared_error(yte, pred)**0.5, 3))
    print("Median AE:", round(median_absolute_error(yte, pred), 3))
    print("MAPE:", round(mean_absolute_percentage_error(yte, pred), 3))
    print("R2:", round(r2_score(yte, pred), 3))

def main():
    X, y = make_regression(n_samples=600, n_features=10, noise=20, random_state=42)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    evaluate("Linear Regression", make_pipeline(StandardScaler(), LinearRegression()),
             Xtr, Xte, ytr, yte)
    evaluate("Random Forest", RandomForestRegressor(n_estimators=150, random_state=42,
             n_jobs=-1), Xtr, Xte, ytr, yte)

if __name__ == "__main__":
    main()
