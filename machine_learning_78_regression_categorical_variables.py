"""
ML 78: Regression with Categorical Variables
Uses OneHotEncoder inside a Pipeline to safely encode categories.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    rng = np.random.default_rng(78)
    n = 260
    experience = rng.integers(0, 21, n)
    education = rng.choice(["Bachelor", "Master", "PhD"], n)
    department = rng.choice(["Sales", "Technology", "Operations"], n)
    education_effect = pd.Series(education).map(
        {"Bachelor": 0, "Master": 9000, "PhD": 16000}
    ).to_numpy()
    department_effect = pd.Series(department).map(
        {"Sales": 3000, "Technology": 7000, "Operations": 0}
    ).to_numpy()
    salary = 30000 + experience * 2400 + education_effect + department_effect
    salary = salary + rng.normal(0, 6500, n)

    X = pd.DataFrame({
        "experience": experience,
        "education": education,
        "department": department,
    })
    y = salary
    categorical = ["education", "department"]
    numeric = ["experience"]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regression", LinearRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=78
    )
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    print("Regression with Categorical Variables")
    print(f"MAE: {mean_absolute_error(y_test, predicted):,.2f}")
    print(f"R²:  {r2_score(y_test, predicted):.3f}")
    example = pd.DataFrame({
        "experience": [5, 12],
        "education": ["Bachelor", "Master"],
        "department": ["Sales", "Technology"],
    })
    print("Example salary predictions:", np.round(model.predict(example), 2))


if __name__ == "__main__":
    main()
