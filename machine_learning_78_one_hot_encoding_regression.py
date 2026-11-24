"""ML 78: Regression with categorical variables using OneHotEncoder."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def main():
    rng = np.random.default_rng(42)
    n = 120
    df = pd.DataFrame({
        "size_sqft": rng.integers(500, 2200, n),
        "city": rng.choice(["Delhi", "Pune", "Jaipur"], n),
        "property_type": rng.choice(["Apartment", "House"], n),
    })
    city_effect = df["city"].map({"Delhi": 30, "Pune": 15, "Jaipur": 5})
    type_effect = df["property_type"].map({"Apartment": 10, "House": 20})
    y = 0.08 * df["size_sqft"] + city_effect + type_effect + rng.normal(0, 8, n)

    prep = ColumnTransformer([
        ("numeric", StandardScaler(), ["size_sqft"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         ["city", "property_type"]),
    ])
    model = make_pipeline(prep, LinearRegression())
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("RMSE:", round(mean_squared_error(y_test, pred) ** 0.5, 2))
    print("First five predictions:", pred[:5].round(2))

if __name__ == "__main__":
    main()
