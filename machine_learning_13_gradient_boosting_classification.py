"""
Machine Learning 13: Gradient Boosting Classification
Train and evaluate a GradientBoostingClassifier on a sample dataset.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Gradient Boosting Classification")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, predictions))
    print("\nClassification report:\n", classification_report(
        y_test, predictions, target_names=data.target_names
    ))


if __name__ == "__main__":
    main()
