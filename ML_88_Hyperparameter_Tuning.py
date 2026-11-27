"""
Machine Learning File 88: Hyperparameter Tuning with GridSearchCV
Finds useful Random Forest settings using cross-validation.
Install: pip install scikit-learn
"""

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report


def main():
    dataset = load_wine()
    X, y = dataset.data, dataset.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    parameter_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 4],
    }

    search = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=parameter_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)

    print("Best parameters:", search.best_params_)
    print(f"Best cross-validation accuracy: {search.best_score_:.3f}")
    print(f"Test accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=dataset.target_names))


if __name__ == "__main__":
    main()
