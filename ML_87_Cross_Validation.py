"""
Machine Learning File 87: K-Fold Cross-Validation
Compares Logistic Regression and Random Forest using stratified folds.
Install: pip install scikit-learn
"""

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main():
    dataset = load_wine()
    X, y = dataset.data, dataset.target

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=42)
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
    }

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        print(f"\n{name}")
        print("Fold scores:", scores.round(3))
        print(f"Mean accuracy: {scores.mean():.3f}")
        print(f"Standard deviation: {scores.std():.3f}")


if __name__ == "__main__":
    main()
