"""ML 80: Train, export, reload, and use a regression model."""
from pathlib import Path
import joblib
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():
    # Use a built-in dataset; no manual download or custom data file required.
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=300, n_features=6, noise=12,
                           random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(X_train, y_train)

    model_path = Path(__file__).with_name("regression_model_80.joblib")
    joblib.dump(model, model_path)
    loaded_model = joblib.load(model_path)
    sample_prediction = loaded_model.predict(X_test[:3])
    print("Saved model to:", model_path)
    print("Reloaded model predictions:", sample_prediction.round(2))
    print("R² score:", round(loaded_model.score(X_test, y_test), 3))

if __name__ == "__main__":
    main()
