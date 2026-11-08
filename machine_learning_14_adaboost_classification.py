"""
Machine Learning 14: AdaBoost Classification
Boost a collection of weak learners and evaluate the classifier.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


def main():
    data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )

    # estimator is the current scikit-learn parameter name.
    weak_learner = DecisionTreeClassifier(max_depth=1, random_state=42)
    model = AdaBoostClassifier(
        estimator=weak_learner, n_estimators=100, learning_rate=0.5, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("AdaBoost Classification")
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print("\nClassification report:\n", classification_report(
        y_test, predictions, target_names=data.target_names
    ))


if __name__ == "__main__":
    main()
