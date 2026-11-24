"""ML 77: Practical feature engineering for regression."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def main():
    # Synthetic mixed-type dataset keeps this example self-contained.
    data = pd.DataFrame({
        "area": [650, 800, 950, 1100, 1250, 1400, 1550, 1700, 1850, 2000],
        "bedrooms": [1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        "neighborhood": ["A", "B", "A", "C", "B", "C", "A", "B", "C", "A"],
        "age": [20, 15, 10, 8, 12, 5, 7, 4, 3, 2],
        "price": [120, 150, 175, 190, 220, 245, 265, 300, 325, 350],
    })
    X, y = data.drop(columns="price"), data["price"]
    numeric = ["area", "bedrooms", "age"]
    categorical = ["neighborhood"]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    model = Pipeline([("prep", prep), ("regressor", Ridge(alpha=1.0))])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")
    print(f"R²: {r2_score(y_test, pred):.3f}")
    print("Predictions:", pred.round(2))

if __name__ == "__main__":
    main()
