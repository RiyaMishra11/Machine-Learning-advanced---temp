"""
ML 77: Feature Engineering for Regression
Demonstrates numerical imputation/scaling and categorical imputation/encoding
with a ColumnTransformer and a regression pipeline.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    rng = np.random.default_rng(77)
    n = 300
    area = rng.normal(1200, 350, n).clip(400, 3000)
    bedrooms = rng.integers(1, 6, n).astype(float)
    location = rng.choice(["Central", "Suburban", "Rural"], n)
    age = rng.integers(0, 40, n).astype(float)
    location_bonus = pd.Series(location).map(
        {"Central": 45000, "Suburban": 20000, "Rural": 0}
    ).to_numpy()
    price = 70000 + area * 180 + bedrooms * 12000 - age * 900 + location_bonus
    price += rng.normal(0, 18000, n)

    X = pd.DataFrame({
        "area": area, "bedrooms": bedrooms, "age": age, "location": location
    })
    y = price
    # Add a few missing values to demonstrate preprocessing.
    X.loc[rng.choice(n, 12, replace=False), "area"] = np.nan
    X.loc[rng.choice(n, 10, replace=False), "location"] = np.nan

    numeric_features = ["area", "bedrooms", "age"]
    categorical_features = ["location"]
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", Ridge(alpha=1.0)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=77
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print("Feature Engineering for Regression")
    print(f"MAE:  {mean_absolute_error(y_test, predictions):,.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, predictions)):,.2f}")
    print(f"R²:   {r2_score(y_test, predictions):.3f}")
    print("Sample predictions:", np.round(predictions[:5], 2))


if __name__ == "__main__":
    main()
